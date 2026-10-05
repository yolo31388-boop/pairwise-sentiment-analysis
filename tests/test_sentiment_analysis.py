"""
游戏内玩家情感分析与情绪识别系统 - 验收测试
运行方式：python -m pytest tests/test_sentiment_analysis.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.sentiment import SentimentAnalysis


class TestSentimentAnalysis:
    def setup_method(self):
        self.system = SentimentAnalysis()

    # ---- analyze_sentiment：维度 + 强度 ----
    def test_case_01(self):
        result = self.system.analyze_sentiment("这游戏好开心，真的很喜欢")
        assert set(result["dimensions"].keys()) == {"joy", "anger", "sadness", "fear"}
        assert result["dimensions"]["joy"] > 0

    def test_case_02(self):
        plain = self.system.analyze_sentiment("好玩")
        strong = self.system.analyze_sentiment("这个游戏真的非常好玩，超级开心")
        assert 0.0 <= plain["intensity"] <= 1.0
        assert strong["intensity"] > plain["intensity"]

    def test_case_03(self):
        result = self.system.analyze_sentiment("气死我了，这游戏太垃圾")
        assert result["dominant_emotion"] == "anger"
        assert result["polarity"] == "negative"
        assert result["compound"] < 0

    def test_case_04(self):
        happy = self.system.analyze_sentiment("玩得好开心，哈哈")
        assert happy["polarity"] == "positive"
        assert happy["compound"] > 0

    # ---- emotion_recognition：特征 + 上下文 ----
    def test_case_05(self):
        result = self.system.emotion_recognition("这游戏太垃圾了")
        words = [f["word"] for f in result["features"]]
        assert "垃圾" in words
        feature = next(f for f in result["features"] if f["word"] == "垃圾")
        assert feature["emotion"] == "anger"
        assert feature["intensifier"] == "太"

    def test_case_06(self):
        assert self.system.emotion_recognition("气死我了")["emotion"] == "anger"
        assert self.system.emotion_recognition("好难过，有点伤心")["emotion"] == "sadness"
        assert self.system.emotion_recognition("这关好恐怖，吓到了")["emotion"] == "fear"
        assert self.system.emotion_recognition("真开心，游戏很棒")["emotion"] == "joy"

    def test_case_07(self):
        result = self.system.emotion_recognition(
            "又来了", context=["这游戏太垃圾了，气死我了"])
        assert result["context_used"] is True
        assert result["emotion"] == "anger"

    def test_case_08(self):
        context_result = self.system.emotion_recognition("气死我了，太垃圾")
        result = self.system.emotion_recognition("再来一次", context=context_result)
        assert result["context_used"] is True
        assert result["emotion"] == "anger"

    # ---- sentiment_trend：时间 + 预测 ----
    def test_case_09(self):
        self.system.record("好玩", timestamp=1.0)
        self.system.record("开心", timestamp=2.0)
        trend = self.system.sentiment_trend()
        assert trend["count"] == 2
        assert trend["points"][0]["timestamp"] < trend["points"][1]["timestamp"]
        assert all("score" in p for p in trend["points"])

    def test_case_10(self):
        self.system.record("好玩，开心", timestamp=1.0)
        self.system.record("一般般", timestamp=2.0)
        self.system.record("有点生气", timestamp=3.0)
        self.system.record("垃圾，气死我了", timestamp=4.0)
        trend = self.system.sentiment_trend()
        assert trend["direction"] == "falling"
        assert trend["slope"] < 0

    def test_case_11(self):
        self.system.record("垃圾，气死", timestamp=1.0)
        self.system.record("好开心，好玩", timestamp=2.0)
        trend = self.system.sentiment_trend()
        assert trend["prediction"] is not None
        assert -1.0 <= trend["prediction"] <= 1.0
        assert trend["prediction"] > trend["points"][-1]["score"]

    # ---- sentiment_alert：阈值 + 分级 ----
    def test_case_12(self):
        self.system.record("有点生气", timestamp=1.0)
        self.system.record("真讨厌", timestamp=2.0)
        self.system.record("好烦啊", timestamp=3.0)
        alert = self.system.sentiment_alert()
        assert alert["triggered"] is True
        assert alert["level"] == "warning"
        assert "warning" in alert["thresholds"] and "critical" in alert["thresholds"]

    def test_case_13(self):
        self.system.record("垃圾，气死我了", timestamp=1.0)
        self.system.record("太垃圾，讨厌死了", timestamp=2.0)
        critical_alert = self.system.sentiment_alert()
        assert critical_alert["level"] == "critical"

        happy_system = SentimentAnalysis()
        happy_system.record("真开心，好玩，很棒", timestamp=1.0)
        happy_system.record("超级喜欢，太爽了", timestamp=2.0)
        ok_alert = happy_system.sentiment_alert()
        assert ok_alert["triggered"] is False
        assert ok_alert["level"] == "none"

    # ---- player_satisfaction：指标 + 改进 ----
    def test_case_14(self):
        self.system.record("垃圾，气死我了", timestamp=1.0)
        self.system.record("好难过，伤心", timestamp=2.0)
        self.system.record("太坑了，好烦", timestamp=3.0)
        satisfaction = self.system.player_satisfaction()
        assert 0.0 <= satisfaction["score"] <= 100.0
        metrics = satisfaction["metrics"]
        assert {"avg_sentiment", "positive_ratio", "negative_ratio",
                "trend_direction", "alert_level"} <= set(metrics.keys())
        assert satisfaction["level"] == "unsatisfied"
        assert len(satisfaction["improvements"]) > 0
        assert isinstance(satisfaction["improvements"][0], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
