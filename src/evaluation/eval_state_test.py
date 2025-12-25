import numpy as np
import pytest

from src.evaluation.state import State
from src.evaluation.eval_state import EvalState 

# 模拟一个简单的 EvalState
@pytest.fixture
def eval_state() -> EvalState:
    labels = np.array([0, 1, 0, 1, 0])  # 真实标签
    predicts = np.array([0, 1, 1, 1, 0])  # 预测结果
    return EvalState(labels=labels, predicts=predicts, total_classes=2)



class TestEvalState:
    """EvalState 单元测试"""

    def test_to_ids_1d(self):
        """labels/predicts 作为 1D 类别 id"""
        labels = np.array([0, 1, 2, 2, 1])
        preds = np.array([0, 2, 2, 1, 1])

        es = EvalState(labels=labels, predicts=preds)
        assert np.all(es.y_true == labels)
        assert np.all(es.y_pred == preds)

    def test_to_ids_2d(self):
        """labels/predicts 为 2D one-hot / logits"""
        labels = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        preds = np.array([
            [0.9, 0.1, 0.0],  # -> 0
            [0.1, 0.8, 0.1],  # -> 1
            [0.2, 0.1, 0.7]   # -> 2
        ])

        es = EvalState(labels=labels, predicts=preds)
        assert np.all(es.y_true == np.array([0, 1, 2]))
        assert np.all(es.y_pred == np.array([0, 1, 2]))
        
    def test_label_to_ids_2d(self):
        """labels/predicts 为 2D one-hot / logits"""
        labels = np.array([
            0, 1, 2
        ])
        preds = np.array([
            [0.9, 0.1, 0.0],  # -> 0
            [0.1, 0.8, 0.1],  # -> 1
            [0.2, 0.1, 0.7]   # -> 2
        ])

        es = EvalState(labels=labels, predicts=preds)
        assert np.all(es.y_true == np.array([0, 1, 2]))
        assert np.all(es.y_pred == np.array([0, 1, 2]))
        

    def test_num_labels_infer_from_2d(self):
        """num_labels 从 2D shape 推断"""
        labels = np.zeros((5, 10))
        preds = np.zeros((5, 10))
        es = EvalState(labels=labels, predicts=preds)

        assert es.num_labels == 10

    def test_num_labels_infer_from_1d(self):
        """num_labels 从最大 label id 推断"""
        labels = np.array([0, 1, 2, 2, 1])
        preds = np.array([0, 2, 2, 1, 1])
        es = EvalState(labels=labels, predicts=preds)

        assert es.num_labels == 3

    def test_num_labels_with_total_classes(self):
        """num_labels 优先使用 total_classes"""
        labels = np.array([0, 1, 2])
        preds = np.array([0, 1, 2])
        es = EvalState(labels=labels, predicts=preds, total_classes=10)

        assert es.num_labels == 10

    def test_confusion_matrix(self):
        """混淆矩阵计算正确"""
        labels = np.array([0, 1, 2, 2, 1])
        preds = np.array([0, 2, 2, 1, 1])
        es = EvalState(labels=labels, predicts=preds)

        cm = es.confusion_matrix
        # cm[i][j] 真实 i 预测 j
        expected = np.array([
            [1, 0, 0],  # true=0 -> pred=0 count=1
            [0, 1, 1],  # true=1 -> pred=1 count=1, pred=2 count=1
            [0, 1, 1]   # true=2 -> pred=1 count=1, pred=2 count=1
        ])
        assert np.all(cm == expected)

    def test_cm_stats(self):
        """_cm_stats 输出 TP/FP/TN/FN 数组正确"""
        labels = np.array([0, 1, 2, 2, 1])
        preds = np.array([0, 2, 2, 1, 1])
        es = EvalState(labels=labels, predicts=preds)

        TP, FP, TN, FN = es._cm_stats

        # 从混淆矩阵 expected 推导：
        # TP = diag = [1, 1, 1]
        assert np.all(TP == np.array([1, 1, 1]))

        # FP(列和 - TP):
        # pred=0: col sum=1 -> FP=0
        # pred=1: col sum=2 -> FP=1
        # pred=2: col sum=2 -> FP=1
        assert np.all(FP == np.array([0, 1, 1]))

        # FN(行和 - TP):
        # true=0 row sum=1 -> FN=0
        # true=1 row sum=2 -> FN=1
        # true=2 row sum=2 -> FN=1
        assert np.all(FN == np.array([0, 1, 1]))

        # TN = total - (TP+FP+FN) total=5
        # class0: 5-(1+0+0)=4
        # class1: 5-(1+1+1)=2
        # class2: 5-(1+1+1)=2
        assert np.all(TN == np.array([4, 2, 2]))

    def test_micro_state(self):
        """micro_state 的 TP/FP/FN/TN 聚合正确"""
        labels = np.array([0, 1, 2, 2, 1])
        preds = np.array([0, 2, 2, 1, 1])
        es = EvalState(labels=labels, predicts=preds)

        ms = es.micro_state
        # 通过上面的混淆矩阵 _cm_stats 得到
        # TP sum=3 FP sum=2 FN sum=2 TN sum=8
        assert ms.TP == 3
        assert ms.FP == 2
        assert ms.FN == 2
        assert ms.TN == 8

        # micro precision = TP/(TP+FP) = 3/5 = 0.6
        assert ms.precision == pytest.approx(0.6)
        # micro recall = TP/(TP+FN) = 3/5 = 0.6
        assert ms.recall == pytest.approx(0.6)
        # micro f1 = 0.6
        assert ms.f1 == pytest.approx(0.6)

    def test_macro_metrics(self):
        """macro precision/recall/f1 应是 per-class 平均"""
        labels = np.array([0, 1, 2, 2, 1])
        preds = np.array([0, 2, 2, 1, 1])
        es = EvalState(labels=labels, predicts=preds)

        # class0: TP=1 FP=0 FN=0 -> P=1 R=1 F1=1
        # class1: TP=1 FP=1 FN=1 -> P=0.5 R=0.5 F1=0.5
        # class2: TP=1 FP=1 FN=1 -> P=0.5 R=0.5 F1=0.5
        # macro = mean => (1 + 0.5 + 0.5)/3 = 0.6666667
        assert es.macro_precision == pytest.approx(2/3)
        assert es.macro_recall == pytest.approx(2/3)
        assert es.macro_f1_score == pytest.approx(2/3)

    def test_state_one_vs_rest(self):
        """state_one_vs_rest(k) 正确返回某个类别的 TP/FP/TN/FN"""
        labels = np.array([0, 1, 2, 2, 1])
        preds = np.array([0, 2, 2, 1, 1])
        es = EvalState(labels=labels, predicts=preds)

        s1 = es.state_one_vs_rest(1)
        assert isinstance(s1, State)
        assert (s1.TP, s1.FP, s1.TN, s1.FN) == (1, 1, 2, 1)

    def test_invalid_shape_raises(self):
        """输入 shape 非 1D/2D 时应抛 ValueError"""
        labels = np.zeros((2, 3, 4))
        preds = np.zeros((2, 3, 4))
        es = EvalState(labels=labels, predicts=preds)

        with pytest.raises(ValueError):
            _ = es.y_true

        with pytest.raises(ValueError):
            _ = es.y_pred


    def test_to_dict_accuracy(self, eval_state: EvalState):
        """测试micro accuracy 是否正确返回"""
        result = eval_state.calculate()
        
        # 正确预测的样本数 / 总样本数
        assert result["acc"] == pytest.approx(4 / 5 * 100, rel=0.001)  

    def test_to_dict_micro_precision(self, eval_state: EvalState):
        """ 测试micro precision 是否正确返回
            NOTE 打印出 TP=4 FP=1 TN=4 FN=1，加起来等于 10，样本只有五个，这看起来很反直觉，但在宏平均逻辑之下是正确的, 
            NOTE 这是必然且正确的，宏平均的逻辑是把每一个类别都当成一次独立的“二分类任务”，然后把所有任务的 TP/FP/TN/FN 加起来, 因此 Micro 指标的统计基数本来就是 样本数 × 类别数，而不是样本数。
        """
        print(eval_state)
        print(eval_state.micro_state)
        
        result = eval_state.calculate()
        assert result["micro"]["p"] == pytest.approx(4 / 5 * 100, rel=0.001)  # precision = TP / (TP + FP)

    def test_to_dict_micro_recall(self, eval_state: EvalState):
        """测试micro recall 是否正确返回"""
        result = eval_state.calculate()
        assert "micro" in result
        assert "r" in result["micro"]
        assert result["micro"]["r"] == pytest.approx(4 / 5 * 100, rel=0.001)  # recall = TP / (TP + FN)


    def test_to_dict_macro_precision(self, eval_state: EvalState):
        """测试macro precision 是否正确返回"""
        result = eval_state.calculate()
        assert "macro" in result
        assert "p" in result["macro"]
        assert result["macro"]["p"] == pytest.approx(5 / 6 * 100, rel=0.001)

    # 宏平均就是把各类的指标算术平均
    def test_to_dict_macro_recall(self, eval_state: EvalState):
        """测试macro recall 是否正确返回"""
        result = eval_state.calculate()
        assert "macro" in result
        assert "r" in result["macro"]
        assert result["macro"]["r"] == pytest.approx(5 / 6 * 100, rel=0.001)
        