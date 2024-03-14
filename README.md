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
[ipynb連結](https://nbviewer.org/github/weiso131/Q_learning/blob/main/Untitled.ipynb)
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


### 結果
基本上都能在個位數個步驟走完，因為場地只有4x4

![image](https://github.com/weiso131/Q_learning/assets/131360912/f5710161-1c0e-49f5-9bda-1a5737d86402)


但有時候會繞路

![image](https://github.com/weiso131/Q_learning/assets/131360912/f6f5a25e-0989-4567-b906-c95cf9921037)

再多訓練久一點，或許真的能訓練出能一直找出最佳解的模型，但那太久了

而且場地再大一點，訓練所需的epoch數量只會更大

直接使用4個狀態的方法不夠有效率

## 寫個腳本
雖然我用Q learning來解這個遊戲

但這遊戲顯而易見的有個很簡單的最佳解腳本
```python=
def action_choice(state):
    py, px, gy, gx = state
    
    if (py != gy):
        if (gy - py < 0):
            return 0
        else:
            return 1
    elif (px != gx):
        if (gx - px < 0):
            return 2
        else:
            return 3
```
### 用腳本替代隨機
[ipynb連結](https://nbviewer.org/github/weiso131/Q_learning/blob/best_solution/q_learning.ipynb)

我延長了參考腳本的時間

到了幾乎由Q_form全權決定行動的50000過後

找到的路徑步數也都壓在8以內

![image](https://github.com/weiso131/Q_learning/assets/131360912/63dbfeac-256c-40cd-a0b9-3887964f545d)


似乎很順利地學到最佳解做法

![image](https://github.com/weiso131/Q_learning/assets/131360912/a65d716b-f6c7-4568-bfe7-6e1a54106887)


我有試過20x20的地圖，訓練過程完全參考腳本

經過5e6個epoch之後，模型也能學到最佳解

看來有個腳本來學習是最有效率的

事實上，在其他的強化學習應用中，一開始先參考純貪心法的腳本似乎也蠻常見的(by chatGPT)

只是當有個腳本是最佳解的時候

還在訓練模型

根本是多此一舉啊

## 調整Q_form
[ipynb連結](https://nbviewer.org/github/weiso131/Q_learning/blob/new_way/q_learning.ipynb)
前面的Q_form是直接接收player和goal的位置，然而，我們其實只要知道他們相減的數值正負就能判斷該往哪移動

於是我就把Q_表格改成這樣:
```python=
Q_form = np.zeros((3, 3) + (4,))
#goal在player上或下或沒差, goal在player左或右或沒差
```
並對state做轉換
```python=
def convert(p, g):
    """
    將玩家與目標的距離是正或負或0
    轉換成0或1或2
    """
    if (g - p > 0):
        return 0
    elif (g - p < 0):
        return 1
    else:
        return 2

def state_convert(state):
    """
    將原本遊戲回報的state
    轉成Q_form可用的格式
    """
    py, px, gy, gx = state
    return (convert(py, gy), convert(px, gx))
```
### 測試結果
回到與最初的狀態一樣，是先隨機亂走，原本也跟之前一樣epoch=5e5，但在把與腳本最佳解的步數差距圖像化之後，發現我其實訓練到20000他們的差距幾乎都是0了(現在才想起來我可以把它圖像化)

#### 訓練過程與腳本最佳解的步數差距
後半部偶爾會出現非最佳解，應該是剛好隨機到隨機亂走的模式(我沒有讓隨機亂走的機率衰減到0)

![image](https://github.com/weiso131/Q_learning/assets/131360912/84e01709-a5c5-49d8-8c6e-5b63541603c5)


#### 測試過程與腳本最佳解的步數差距
他看起來是學到最佳解的方法了

![image](https://github.com/weiso131/Q_learning/assets/131360912/4f060255-c09b-4b20-ae96-54f7e500e450)




#### 換成100x100的地圖，並且跑10000次
因為這個方法只關心相對位置，所以不管幾乘幾都能找到最佳解

![image](https://github.com/weiso131/Q_learning/assets/131360912/6de5e217-0cda-46e2-a208-8f2fdd1311fa)

