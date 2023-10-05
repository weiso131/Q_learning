import numpy as np

class game():
    def __init__(self, n, m):
        self.player = [0, 0]
        self.goal = [n - 1, m - 1]
        self.n = n#遊戲地圖y方向長度
        self.m = m#遊戲地圖x方向長度
        self.maps = []
        self.move = ((-1, 0), (1, 0), (0, -1), (0, 1))#上,下,左,右
        self.clear_maps()
    
        self.start()#在創建遊戲的時候初始化地圖
            
    def random_position(self):
        return [np.random.randint(0, self.n), np.random.randint(0, self.m)]
    
    def start(self):
        
        
        """
        初始化地圖
        """
        
        self.player = self.random_position()
        self.goal = self.random_position()
        
        if (self.player == self.goal):
            self.start()
    
    def clear_maps(self):
        """
        重製地圖
        """
        for i in range(self.n):
            self.maps.append(list(range(self.m)))
            for j in range(self.m):
                self.maps[i][j] = "_"
    
    def update_position(self):
        """
        更新玩家與目標的位置
        """
        self.maps[self.goal[0]][self.goal[1]] = "G"
        self.maps[self.player[0]][self.player[1]] = "P"
        
        
    def show_game(self):
        """
        顯示遊戲狀態
        """
        self.clear_maps()
        self.update_position()
        for i in range(self.n):
            for j in range(self.m):
                print(self.maps[i][j], end = '')
            print()
            
    def player_move(self, oper):
        
        """
        return reward, observation, done
        """
        
        """
        更新位置
        """
        self.player[0] = max(0, min(self.n - 1, self.player[0] + self.move[oper][0]))
        self.player[1] = max(0, min(self.m - 1, self.player[1] + self.move[oper][1]))
        
        """
        如果到達目標
        """
        
        
        if (self.player == self.goal):
            return self.caculate_reward(), self.observation(), True
        
        return self.caculate_reward(), self.observation(), False
    

    def caculate_reward(self):
        """
        回報獎勵值
        """
        return self.n - abs(self.player[0] - self.goal[0]) + \
            self.m - abs(self.player[1] - self.goal[1])

    def observation(self):
        """
        回報狀態
        """
        return (self.player[0], self.player[1], self.goal[0], self.goal[1])
    

"""

new_game = game(10, 20)
new_game.show_game()
while (keyboard.read_key() != 'e'):
    if (keyboard.read_key() == 'w'):
        new_game.player_move(0)
    if (keyboard.read_key() == 's'):
        new_game.player_move(1)
    if (keyboard.read_key() == 'a'):
        new_game.player_move(2)
    if (keyboard.read_key() == 'd'):
        new_game.player_move(3)
    new_game.show_game()
"""
    





        