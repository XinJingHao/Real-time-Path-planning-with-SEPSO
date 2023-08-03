import os
import time


# No Boostrap
os.system('python SEPSO_Up.py --BF 1 --Bootstrap False')
time.sleep(10)

os.system('python SEPSO_Up.py --BF 2 --Bootstrap False')
time.sleep(10)

os.system('python SEPSO_Up.py --BF 3 --Bootstrap False')
time.sleep(10)

os.system('python SEPSO_Up.py --BF 4 --Bootstrap False')
time.sleep(10)

# 评估SEPSO为各个BF找到的Tbest.pt
os.system('python Find_Tbest_params.py')
time.sleep(10)

# 画SEPSO_Up的Fitness曲线
os.system('python Draw_TbestValue_Up.py')
time.sleep(10)
