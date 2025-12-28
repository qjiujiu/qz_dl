import json
from collections import UserDict, defaultdict
from typing import Any, Union, List, Optional

class KVTree(UserDict):
    """ 
    IndexedKVTree: 一个支持模糊路径搜索的高性能嵌套字典包装器。
    
    特性:
    1. 倒排索引: 初始化时构建 Key->Value 索引，单 Key 查找复杂度 O(1)。
    2. 模糊路径: 支持 'A.B' 语法，表示在 A 的子树下搜索 B (中间可跨越任意层级)。
    3. 智能展开: 自动递归解析内部的 JSON 字符串。
    4. 快速失败: 优化了 JSON 解析前的预判逻辑，大幅减少无效 try-catch 开销。
    """

    def __init__(self, data: Union[str, list, dict], sep: str = '.'):
        self.sep = sep
        # 递归展开数据 (JSON 字符串 -> Dict/List)
        expanded_data = self._expand(data)
        
        # 初始化 UserDict，数据存储在 self.data
        super().__init__(expanded_data)
        
        # 构建索引: key -> list of values
        # 如果 key 唯一，列表长度等于 1, 若有重名 key，列表存储所有对应值
        self._index = defaultdict(list)
        self._build_index(self.data)

    def _expand(self, data: Any) -> Any:
        """ 递归展开 JSON 字符串 (包含快速失败优化) """
        if isinstance(data, str):
            # 去除首尾空白，提升判断准确度
            s_data = data.strip()
            
            # 优化核心：只有看起来像 JSON ({} 或 []) 的才尝试解析
            # 这能过滤掉 99% 的普通字符串，避免昂贵的异常捕获开销
            if (s_data.startswith("{") and s_data.endswith("}")) or \
               (s_data.startswith("[") and s_data.endswith("]")):
                try:
                    parsed = json.loads(s_data)
                    # 只有解析结果是结构化数据才继续展开
                    if isinstance(parsed, (dict, list)):
                        return self._expand(parsed)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    pass
            
            # 无法解析或不符合条件的字符串，原样返回
            return data

        elif isinstance(data, dict):
            return {k: self._expand(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._expand(i) for i in data]
        
        return data

    def _build_index(self, current_data: Any):
        """ 一次性遍历（DFS），构建倒排索引。复杂度：O(N)，
            仅在初始化时发生一次。
        """
        if isinstance(current_data, dict):
            for k, v in current_data.items():
                self._index[k].append(v) # 记录索引
                self._build_index(v)     # 递归构建
        elif isinstance(current_data, list):
            for item in current_data:
                self._build_index(item)

    def __getitem__(self, key: str) -> Any:
        """ 
        查找逻辑：
        1. 如果 key 没有分隔符 -> 直接查索引 O(1)。
        2. 如果 key 有分隔符 (A.B) -> 先查 A，再在 A 查询结果之中递归查询 B, 以此类推
        """
        
        if self.sep not in key:
            candidates = self._index.get(key)
            if not candidates:
                raise KeyError(f"Key '{key}' not found.")
            return candidates[0]
        
        # 路径查找 (A.B.C) ---
        parts = key.split(self.sep)
        root_key = parts[0]
        
        # 查找找到路径起点的所有候选对象
        candidates = self._index.get(root_key)
        if not candidates:
             raise KeyError(f"Root key '{root_key}' not found.")
        
        current_candidates = candidates
        
        # 逐级向下搜索
        for part in parts[1:]:
            next_candidates = []
            for scope in current_candidates:
                # 调用 _find_in_scope 进行递归搜索
                found = self._find_in_scope(scope, part)
                if found is not None:
                    next_candidates.append(found)
            
            if not next_candidates:
                raise KeyError(f"Path segment '{part}' not found under '{parts[0]}'")
            current_candidates = next_candidates

        return current_candidates[0]

    def _find_in_scope(self, scope, target_key):
        """ 
        在指定范围内递归查找 key (DFS)。
        实现了 '模糊' 查找：即使 target_key 藏在深层也能找到。
        """
        if isinstance(scope, dict):
            # 检查当前层级
            if target_key in scope:
                return scope[target_key]
            # 递归检查子层级 (这是实现模糊查找的关键)
            for v in scope.values():
                res = self._find_in_scope(v, target_key)
                if res is not None: 
                    return res
        elif isinstance(scope, list):
            # 如果是列表，遍历列表项继续找
            for item in scope:
                res = self._find_in_scope(item, target_key)
                if res is not None: 
                    return res
        return None
    
    def get(self, key, default=None):
        """ 重写 get 方法，复用 __getitem__ 的逻辑 """
        try:
            return self[key]
        except KeyError:
            return default

    def refresh(self):
        """ 如果修改了数据，需要手动重建索引 """
        self._index.clear()
        self._build_index(self.data)
        
    def to_json(self):
        return json.dumps(self.data, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    data = {
        "section_A": {
            "useless_layer": {
                "wrapper": {
                    "target_value": 100
                }
            },
            "strict_child": 200
        },
        "logs": '[{"id": 1, "details": {"nested_json": "true"}}]'
    }

    tree = KVTree(data)

    # 这里 section_A 和 target_value 中间隔了两层，依然能找到
    print("模糊路径测试 (跨越 useless_layer 和 wrapper):")
    print(f"section_A.target_value -> {tree['section_A.target_value']}") 
    
    print("JSON 自动展开测试:")
    print(f"nested_json -> {tree['nested_json']}")

    # logs 列表 -> 第一个元素 -> details -> nested_json
    print("混合测试 (路径 + JSON 内部):")
    print(f"logs.nested_json -> {tree['logs.nested_json']}")