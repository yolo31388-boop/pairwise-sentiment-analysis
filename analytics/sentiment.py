"""
游戏内玩家情感分析与情绪识别系统 - 核心模块

功能：
- analyze_sentiment()   情感分析：多维度（快乐/愤怒/悲伤/恐惧）+ 强度
- emotion_recognition() 情绪识别：特征词提取 + 上下文感知
- sentiment_trend()     情感趋势：时间序列 + 走向预测
- sentiment_alert()     情感预警：阈值 + 分级
- player_satisfaction() 玩家满意度：指标 + 改进建议
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"warning": -0.2, "critical": -0.5}
    )
    trend_window: int = 10


class SentimentAnalysis:
    # 情绪词典：各维度情绪特征词
    EMOTION_LEXICON = {
        "joy": ["开心", "高兴", "快乐", "好玩", "喜欢", "棒", "赞", "爽",
                "有趣", "爱", "哈哈", "不错", "好评", "优秀", "完美"],
        "anger": ["生气", "愤怒", "气死", "垃圾", "坑", "讨厌", "烦", "骂",
                  "滚", "差评", "火大", "可恶", "混蛋"],
        "sadness": ["难过", "伤心", "失望", "哭", "悲伤", "沮丧", "郁闷",
                    "遗憾", "心酸", "委屈"],
        "fear": ["害怕", "恐怖", "吓", "恐惧", "担心", "紧张", "慌", "惊悚"],
    }
    # 程度副词 -> 强度系数
    INTENSIFIERS = {
        "非常": 1.6, "特别": 1.5, "超级": 1.6, "太": 1.4, "很": 1.3,
        "真": 1.2, "好": 1.2, "有点": 0.7, "稍微": 0.5, "略": 0.6,
    }
    # 否定词
    NEGATIONS = ["不", "没", "没有", "无", "别", "未"]
    # 否定后的情绪映射（对立维度）
    OPPOSITE = {"joy": "sadness", "sadness": "joy",
                "anger": "joy", "fear": "joy"}

    DEFAULT_ALERT_THRESHOLDS = {"warning": -0.2, "critical": -0.5}

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _extract_features(self, text: str) -> List[Dict]:
        """从文本中提取情绪特征词，识别程度副词与否定词。"""
        features = []
        if not text:
            return features
        for emotion, words in self.EMOTION_LEXICON.items():
            for word in words:
                start = 0
                while True:
                    idx = text.find(word, start)
                    if idx < 0:
                        break
                    prefix = text[max(0, idx - 3):idx]
                    intensifier = None
                    weight = 1.0
                    for word_mod, factor in self.INTENSIFIERS.items():
                        if word_mod in prefix:
                            intensifier = word_mod
                            weight = factor
                            break
                    negated = any(neg in prefix for neg in self.NEGATIONS)
                    features.append({
                        "word": word,
                        "emotion": emotion,
                        "position": idx,
                        "intensifier": intensifier,
                        "negated": negated,
                        "weight": weight,
                    })
                    start = idx + len(word)
        features.sort(key=lambda f: f["position"])
        return features

    def _dimension_scores(self, features: List[Dict]) -> Dict[str, float]:
        """根据特征词计算各情绪维度得分（0~1）。"""
        raw = {emotion: 0.0 for emotion in self.EMOTION_LEXICON}
        for feat in features:
            target = feat["emotion"]
            weight = feat["weight"]
            if feat["negated"]:
                target = self.OPPOSITE.get(target, target)
            raw[target] += weight
        return {emotion: min(1.0, score / 3.0) for emotion, score in raw.items()}

    @staticmethod
    def _compound_score(dimensions: Dict[str, float]) -> float:
        """由维度得分计算综合情感极性得分（-1~1）。"""
        positive = dimensions.get("joy", 0.0)
        negative = (dimensions.get("anger", 0.0)
                    + dimensions.get("sadness", 0.0)
                    + dimensions.get("fear", 0.0))
        return max(-1.0, min(1.0, positive - negative))

    # ------------------------------------------------------------------
    # 情感分析：维度 + 强度
    # ------------------------------------------------------------------
    def analyze_sentiment(self, text: str) -> Dict:
        """多维度情感分析，返回各情绪维度得分与整体强度。"""
        features = self._extract_features(text)
        dimensions = self._dimension_scores(features)
        intensity = min(1.0, sum(f["weight"] for f in features) / 3.0)
        compound = self._compound_score(dimensions)
        if compound > 0.05:
            polarity = "positive"
        elif compound < -0.05:
            polarity = "negative"
        else:
            polarity = "neutral"
        dominant = max(dimensions, key=dimensions.get)
        if dimensions[dominant] <= 0.0:
            dominant = "neutral"
        return {
            "text": text,
            "dimensions": dimensions,
            "intensity": intensity,
            "compound": compound,
            "polarity": polarity,
            "dominant_emotion": dominant,
        }

    # ------------------------------------------------------------------
    # 情绪识别：特征 + 上下文
    # ------------------------------------------------------------------
    def emotion_recognition(self, text: str,
                            context: Optional[object] = None) -> Dict:
        """识别具体情绪（非简单正负二分），支持特征词与上下文。"""
        analysis = self.analyze_sentiment(text)
        features = self._extract_features(text)
        dimensions = dict(analysis["dimensions"])
        context_used = False

        if context is not None:
            context_emotion = self._resolve_context_emotion(context)
            if context_emotion in dimensions:
                context_used = True
                # 上下文情绪对当前判断产生惯性影响
                dimensions[context_emotion] = min(
                    1.0, dimensions[context_emotion] + 0.3)

        emotion = max(dimensions, key=dimensions.get)
        if dimensions[emotion] <= 0.0:
            emotion = "neutral"
            confidence = 1.0 - analysis["intensity"]
        else:
            confidence = min(1.0, 0.4 + dimensions[emotion])
        return {
            "text": text,
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "features": features,
            "dimensions": dimensions,
            "context_used": context_used,
        }

    def _resolve_context_emotion(self, context: object) -> Optional[str]:
        """从上下文中解析主导情绪。支持字符串、字典或历史记录列表。"""
        if isinstance(context, str):
            return self.analyze_sentiment(context)["dominant_emotion"]
        if isinstance(context, dict):
            if "emotion" in context:
                return context["emotion"]
            if "dominant_emotion" in context:
                return context["dominant_emotion"]
        if isinstance(context, (list, tuple)) and context:
            emotions = [self._resolve_context_emotion(item)
                        for item in context]
            emotions = [e for e in emotions
                        if e in self.EMOTION_LEXICON]
            if emotions:
                return max(set(emotions), key=emotions.count)
        return None

    # ------------------------------------------------------------------
    # 历史记录
    # ------------------------------------------------------------------
    def record(self, text: str,
               timestamp: Optional[float] = None) -> Dict:
        """分析一条玩家文本并写入情感历史（供趋势/预警/满意度使用）。"""
        analysis = self.analyze_sentiment(text)
        entry = {
            "timestamp": timestamp if timestamp is not None else time.time(),
            "text": text,
            "compound": analysis["compound"],
            "polarity": analysis["polarity"],
            "dominant_emotion": analysis["dominant_emotion"],
            "intensity": analysis["intensity"],
        }
        self._history.append(entry)
        return entry

    # ------------------------------------------------------------------
    # 情感趋势：时间 + 预测
    # ------------------------------------------------------------------
    def sentiment_trend(self,
                        history: Optional[List[Dict]] = None) -> Dict:
        """基于时间序列分析情感变化趋势，并预测下一步走向。"""
        records = history if history is not None else self._history
        records = sorted(records, key=lambda r: r["timestamp"])
        points = [{"timestamp": r["timestamp"], "score": r["compound"]}
                  for r in records]
        scores = [p["score"] for p in points]

        slope = 0.0
        if len(points) >= 2:
            slope = self._linear_slope(
                [p["timestamp"] for p in points], scores)

        if slope > 1e-6:
            direction = "rising"
        elif slope < -1e-6:
            direction = "falling"
        else:
            direction = "stable"

        prediction = None
        if points:
            last = points[-1]
            if len(points) >= 2:
                dt = last["timestamp"] - points[-2]["timestamp"]
            else:
                dt = 1.0
            prediction = max(-1.0, min(1.0,
                                       last["score"] + slope * dt))

        return {
            "points": points,
            "count": len(points),
            "direction": direction,
            "slope": slope,
            "prediction": prediction,
        }

    @staticmethod
    def _linear_slope(xs: List[float], ys: List[float]) -> float:
        """最小二乘法拟合斜率（情感随时间的变化率）。"""
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x == 0:
            # 时间戳相同则退化为按序号拟合
            xs = list(range(n))
            mean_x = sum(xs) / n
            var_x = sum((x - mean_x) ** 2 for x in xs)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        return cov / var_x

    # ------------------------------------------------------------------
    # 情感预警：阈值 + 分级
    # ------------------------------------------------------------------
    def sentiment_alert(self,
                        history: Optional[List[Dict]] = None) -> Dict:
        """当近期玩家情感低于阈值时触发分级预警。"""
        thresholds = dict(self.DEFAULT_ALERT_THRESHOLDS)
        thresholds.update(self.config.get("alert_thresholds", {}))
        window = self.config.get("alert_window", 5)

        records = history if history is not None else self._history
        recent = sorted(records, key=lambda r: r["timestamp"])[-window:]

        if not recent:
            return {
                "triggered": False,
                "level": "none",
                "score": None,
                "thresholds": thresholds,
                "message": "暂无玩家情感数据",
            }

        score = sum(r["compound"] for r in recent) / len(recent)
        if score <= thresholds["critical"]:
            level = "critical"
        elif score <= thresholds["warning"]:
            level = "warning"
        else:
            level = "none"

        messages = {
            "critical": "玩家情绪严重负面，需立即介入处理",
            "warning": "玩家情绪偏低，建议关注并安抚",
            "none": "玩家情绪正常",
        }
        return {
            "triggered": level != "none",
            "level": level,
            "score": score,
            "thresholds": thresholds,
            "message": messages[level],
        }

    # ------------------------------------------------------------------
    # 玩家满意度：指标 + 改进
    # ------------------------------------------------------------------
    def player_satisfaction(self,
                            history: Optional[List[Dict]] = None) -> Dict:
        """基于多维指标计算玩家满意度得分（0~100）并给出改进建议。"""
        records = history if history is not None else self._history
        if not records:
            return {
                "score": None,
                "level": "unknown",
                "metrics": {},
                "improvements": ["暂无玩家反馈数据，需先收集玩家情感数据"],
            }

        total = len(records)
        avg_compound = sum(r["compound"] for r in records) / total
        positive_ratio = sum(1 for r in records
                             if r["polarity"] == "positive") / total
        negative_ratio = sum(1 for r in records
                             if r["polarity"] == "negative") / total
        avg_intensity = sum(r["intensity"] for r in records) / total
        trend = self.sentiment_trend(records)
        alert = self.sentiment_alert(records)

        metrics = {
            "avg_sentiment": round((avg_compound + 1.0) / 2.0, 4),
            "positive_ratio": round(positive_ratio, 4),
            "negative_ratio": round(negative_ratio, 4),
            "avg_intensity": round(avg_intensity, 4),
            "trend_direction": trend["direction"],
            "alert_level": alert["level"],
        }

        score = 100.0 * (
            0.4 * metrics["avg_sentiment"]
            + 0.3 * positive_ratio
            + 0.2 * (1.0 - negative_ratio)
            + 0.1 * {"rising": 1.0, "stable": 0.6, "falling": 0.2}[
                trend["direction"]]
        )
        score = round(max(0.0, min(100.0, score)), 2)

        if score >= 75:
            level = "satisfied"
        elif score >= 50:
            level = "neutral"
        else:
            level = "unsatisfied"

        improvements = []
        if negative_ratio > 0.3:
            improvements.append("负面反馈占比过高，优先排查玩家集中抱怨的问题")
        if metrics["avg_sentiment"] < 0.5:
            improvements.append("整体情感倾向偏负面，建议优化被吐槽的玩法与体验")
        if trend["direction"] == "falling":
            improvements.append("玩家情绪呈下降趋势，需及时干预防止流失")
        if alert["level"] == "critical":
            improvements.append("已触发严重预警，立即安排客服介入安抚玩家")
        elif alert["level"] == "warning":
            improvements.append("已触发轻度预警，建议主动回访情绪低落的玩家")
        if avg_intensity < 0.2:
            improvements.append("玩家情绪表达平淡，可通过活动与奖励提升参与感")
        if not improvements:
            improvements.append("玩家满意度良好，保持当前运营策略并持续监控")

        return {
            "score": score,
            "level": level,
            "metrics": metrics,
            "improvements": improvements,
        }
