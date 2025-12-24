from src.evaluation.state import State
import pytest


class TestState:
    """State 指标计算单元测试"""

    def test_precision_normal(self):
        # TP=8 FP=2 => precision=0.8
        s = State(TP=8, FP=2)
        assert s.precision == pytest.approx(0.8)
        assert isinstance(s.precision, float)

    def test_recall_normal(self):
        # TP=6 FN=4 => recall=0.6
        s = State(TP=6, FN=4)
        assert s.recall == pytest.approx(0.6)
        assert isinstance(s.recall, float)

    def test_f1_normal(self):
        # TP=6 FP=2 FN=4
        # precision = 6/8 = 0.75
        # recall = 6/10 = 0.6
        # f1 = 2 * 0.75 * 0.6 / (0.75 + 0.6) = 0.6666667
        s = State(TP=6, FP=2, FN=4)
        assert s.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
        assert isinstance(s.f1, float)

    def test_accuracy_normal(self):
        # TP=7 TN=3 FP=1 FN=1 => accuracy=(7+3)/12=0.8333333
        s = State(TP=7, TN=3, FP=1, FN=1)
        assert s.accuracy == pytest.approx(10 / 12)
        assert isinstance(s.accuracy, float)

    def test_all_zero_safe(self):
        """所有值为 0 不应报错，并且返回 0.0"""
        s = State(TP=0, TN=0, FP=0, FN=0)
        assert s.precision == pytest.approx(0.0)
        assert s.recall == pytest.approx(0.0)
        assert s.f1 == pytest.approx(0.0)
        assert s.accuracy == pytest.approx(0.0)

    def test_precision_zero_denominator(self):
        """TP+FP=0 的情况 precision = 0"""
        s = State(TP=0, FP=0, FN=10, TN=10)
        assert s.precision == pytest.approx(0.0)

    def test_recall_zero_denominator(self):
        """TP+FN=0 的情况 recall = 0"""
        s = State(TP=0, FN=0, FP=10, TN=10)
        assert s.recall == pytest.approx(0.0)

    def test_f1_when_precision_recall_zero(self):
        """precision=0 且 recall=0 时 f1 应为 0"""
        s = State(TP=0, FP=0, FN=0, TN=10)
        assert s.f1 == pytest.approx(0.0)

    def test_metrics_monotonicity(self):
        """简单性质：TP 增加，precision/recall 不应下降（在固定 FP/FN 下）"""
        s1 = State(TP=5, FP=5, FN=5)
        s2 = State(TP=10, FP=5, FN=5)

        assert s2.precision > s1.precision
        assert s2.recall > s1.recall
        assert s2.f1 > s1.f1
