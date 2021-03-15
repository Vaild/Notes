蓝桥杯python准备



## 库学习

### datetime

calendar



### time

```python
"""
这里有三个函数 
time.time  是获取当前距离标准时间秒数
time.loacltime  取得秒数后转为 time_struct 结构
time.asctime  将结构转为正常的时间形式
"""
t = time.asctime(time.loacaltime(time.time()))

# time.mktime 将时间转为秒
time.mktime(time.localtime())

# 'Mon Mar 15 15:00:27 2021'
time.ctime() # 将秒数变为字符串

# 看这一天是星期几
datetime.date(2021, 3, 14).weekday()

# 看每月的第一天是周几
for i in range(1, 13):
    print('2021.{}.1'.format(i), '是星期{}。'.format(datetime.date(2021, i, 1).weekday() + 1))
    
# 输出某两个日期之间的所有日期
def datelist2(start, end):
    startdate = datetime.datetime.strptime(start, '%Y-%m-%d')
    enddate = datetime.datetime.strptime(end, '%Y-%m-%d')
    print(startdate)
    datelist = []
    run = startdate
    datelist.append(run)
    while run != enddate:
        run = run + datetime.timedelta(days=1)
        datelist.append(run)
    print(datelist)
    return datelist
```





## 链接

[github-这是往年的赛题python实现](https://github.com/PlutoaCharon/LanQiaoCode_Python)

[CSDN链接](https://blog.csdn.net/Harry______/article/details/109683177)

[博客-链接](https://plutoacharon.github.io/2020/02/23/Python%E7%AE%97%E6%B3%95%E5%AD%A6%E4%B9%A0-%E8%93%9D%E6%A1%A5%E6%9D%AF%E5%AE%98%E6%96%B9%E7%9C%81%E8%B5%9B%E7%9C%9F%E9%A2%98-%E6%8C%81%E7%BB%AD%E6%9B%B4%E6%96%B0/)