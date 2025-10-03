from crewai.tools import BaseTool
from typing import List, Dict
import random

class GenesisTool(BaseTool):
    name: str = "创世纪模拟器"
    description: str = "从细胞自动机模拟生命。输入：initial_seed (整数)。输出：evolved_entities (字典列表：{'type': str, 'strength': int})。"

    def _run(self, initial_seed: int) -> List[Dict]:
        grid = [[0 for _ in range(5)] for _ in range(5)]
        grid[2][2] = initial_seed % 2 + 1
        for gen in range(5):
            new_grid = [[0 for _ in range(5)] for _ in range(5)]
            for i in range(5):
                for j in range(5):
                    neighbors = sum(grid[x][y] for x in range(max(0, i-1), min(5, i+2))
                                    for y in range(max(0, j-1), min(5, j+2)) if (x, y) != (i, j))
                    if grid[i][j] > 0:
                        new_grid[i][j] = 1 if 2 <= neighbors <= 3 else 0
                    else:
                        new_grid[i][j] = 1 if neighbors == 3 else 0
            grid = new_grid
        entities = [{'type': '有机' if cell == 1 else '无机', 'strength': random.randint(1, 10)} for row in grid for cell in row if cell > 0]
        return entities