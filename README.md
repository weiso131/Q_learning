# Q_learning


## Q learning在幹嘛
相較神經網路是"算出"答案
q_learing比較像是在查表
利用回傳狀態的值來當作index(如果是連續的數值要做離散化)呼叫Q learning表格的數值
對應的數值是做某個action可能會拿到的reward
可想而知，某個狀態對應的Q form有最大可能reward的動作
將會是我們接下來會做的動作
```python=
Q_form = np.zeros((n, m, n, m) + (4,))
"""
前四個就是回傳的狀態對應的維度
最後一個代表有4種動作
"""
```

## Q learning怎麼學習

![image](https://github.com/weiso131/Q_learning/assets/131360912/05da2efa-6410-4d69-8da9-244beb96bee0)



[維基百科](https://zh.wikipedia.org/zh-tw/Q%E5%AD%A6%E4%B9%A0)

### 概念
Q_form上個狀態執行某個action的可能reward值將會等於
(1 - learning rate) * 原本的值 + 
learing rate * (reward + 常數 * 執行action後的狀態可能拿到的最大reward)

### python實作

```python=
Q_form[old_state + (action,)] = (1 - lr) * Q_form[old_state + (action,)] + \
                                        lr * (reward + gamma * np.max(Q_form[new_state]))
"""
lr是learning rate
action是剛執行的動作
old_state是導出action這個動作的狀態
new_state是執行action之後的狀態
reward就...reward
gamma是discount factor，代表要對未來可能獲得的reward之重視程度
"""
```




## 遊戲規則說明
![image](https://github.com/weiso131/Q_learning/assets/131360912/a8103c16-4800-439a-ad47-ab873bcd67aa)
- 用m和n決定遊戲場地的大小
- 'P'是player
- 'G'是goal
- '_'是空地
- 移動P到G來得分

## game的函數操作
### 初始化遊戲
```python=
from self_do_pygame import game
n = 4
m = 4
new_game = game(n, m)
```
### 印出遊戲畫面
```python=
new_game.show_game()
```
### 回傳狀態
```python=
py, px, gy, gx = new_game.observation())
"""
py是player的縱座標值
px是player的橫座標值
gy是goal的縱座標值
gx是goal的橫座標值
"""
```
### 移動
```python=
reward, observation, done = new_game.player_move(action)
"""
action的值可以是0, 1, 2, 3
分別對應上下左右

new_game.player_move會回傳三個資訊
reward      -> 40 - abs(px - gx) - abs(py - gy)
observation -> 同new_game.observation()
done        -> player的座標是否和goal重疊(是否得分)
"""
```

## 最初的想法
- 直接使用observation的四個值來呼叫Q_learning的表格
- 一開始先四個方向隨機亂走，之後再使用Q_learning表格的數值
- 然後我train了500000次

### 訓練過程
#### 隨機成分較高的時候
剛開始隨機亂走比較多，但因為場地只有4x4，player勉強能走到goal

![image](https://github.com/weiso131/Q_learning/assets/131360912/96496dd8-6c8c-4f06-8685-4506f73893e5)


#### 隨機剛開始變少的時候
依照表格做事，結果經常找不到goal

![image](https://github.com/weiso131/Q_learning/assets/131360912/9cc592ac-07bd-43cd-80bd-27b2d94103d2)


#### 一段時間之後
慢慢能找到通往目標的路

![image](https://github.com/weiso131/Q_learning/assets/131360912/a00c1e47-2a7d-47b0-ad74-f61c6f209af9)

![image](https://github.com/weiso131/Q_learning/assets/131360912/59003c31-b3ac-4be5-b1e8-994c0f22f45f)


#### epoch超過1e6之後
基本上都能找到通往目標的路，之後到5e6都是在慢慢優化

![image](https://github.com/weiso131/Q_learning/assets/131360912/130eff2d-41eb-4d36-8fab-7fd8c1835696)


## 結果
基本上都能在個位數個步驟走完，因為場地只有4x4

![image](https://github.com/weiso131/Q_learning/assets/131360912/f5710161-1c0e-49f5-9bda-1a5737d86402)


但有時候會繞路

![image](https://github.com/weiso131/Q_learning/assets/131360912/f6f5a25e-0989-4567-b906-c95cf9921037)

再多訓練久一點，或許真的能訓練出能一直找出最佳解的模型，但那太久了
而且場地再大一點，訓練所需的epoch數量只會更大
直接使用4個狀態的方法不夠有效率


