import ast
from typing import Optional, List
try:
    import networkx as nx
except Exception:
    nx = None

class RUBYMetric:
    def __init__(self):
        self.available_representations = ['PDG', 'AST', 'text']

    def compute_ruby(self, reference_code: str, translated_code: str) -> float:
        # Try GRS (PDG level)
        grs_score = self.compute_grs(reference_code, translated_code)
        if grs_score is not None:
            return grs_score

        # Try TRS (AST level)
        trs_score = self.compute_trs(reference_code, translated_code)
        if trs_score is not None:
            return trs_score

        # Fallback to STS (string/token level)
        return self.compute_sts(reference_code, translated_code)

    def compute_grs(self, reference_code: str, translated_code: str) -> Optional[float]:
        if nx is None:
            return None
        try:
            pdg_ref = self.build_pdg(reference_code)
            pdg_trans = self.build_pdg(translated_code)
            if pdg_ref is None or pdg_trans is None:
                return None
            ged = self.graph_edit_distance(pdg_ref, pdg_trans)
            pdg_size = len(pdg_ref.nodes) + len(pdg_ref.edges) + len(pdg_trans.nodes) + len(pdg_trans.edges)
            return 1.0 - (ged / pdg_size) if pdg_size > 0 else 0.0
        except Exception:
            return None

    def compute_trs(self, reference_code: str, translated_code: str) -> Optional[float]:
        try:
            ast_ref = self.parse_ast(reference_code)
            ast_trans = self.parse_ast(translated_code)
            if ast_ref is None or ast_trans is None:
                return None
            ted = self.tree_edit_distance(ast_ref, ast_trans)
            tree_size = self.count_ast_nodes(ast_ref) + self.count_ast_nodes(ast_trans)
            return 1.0 - (ted / tree_size) if tree_size > 0 else 0.0
        except Exception:
            return None

    def compute_sts(self, reference_code: str, translated_code: str) -> float:
        tokens_ref = self.tokenize_code(reference_code)
        tokens_trans = self.tokenize_code(translated_code)
        sed = self.string_edit_distance(tokens_ref, tokens_trans)
        max_length = max(len(tokens_ref), len(tokens_trans))
        return 1.0 - (sed / max_length) if max_length > 0 else 1.0

    def build_pdg(self, code: str):
        try:
            tree = self.parse_ast(code)
            if tree is None:
                return None
            pdg = nx.DiGraph()
            # simplified: add one node per ast node
            for i, node in enumerate(ast.walk(tree)):
                pdg.add_node(i, type=type(node).__name__)
            return pdg
        except Exception:
            return None

    def parse_ast(self, code: str):
        try:
            return ast.parse(code)
        except SyntaxError:
            return None

    def tokenize_code(self, code: str) -> List[str]:
        tokens = []
        current_token = ""
        for char in code:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in '(){}[];.,=+-*/<>!&|':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
        if current_token:
            tokens.append(current_token)
        return tokens

    def string_edit_distance(self, tokens1: List[str], tokens2: List[str]) -> int:
        m, n = len(tokens1), len(tokens2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    def tree_edit_distance(self, tree1, tree2) -> int:
        return abs(self.count_ast_nodes(tree1) - self.count_ast_nodes(tree2))

    def count_ast_nodes(self, node) -> int:
        count = 1
        for child in ast.iter_child_nodes(node):
            count += self.count_ast_nodes(child)
        return count

    def graph_edit_distance(self, graph1, graph2) -> int:
        return abs(len(graph1.nodes) - len(graph2.nodes)) + abs(len(graph1.edges) - len(graph2.edges))
