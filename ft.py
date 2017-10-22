import numpy as np
import matplotlib.pyplot as plt
#x = np.linspace(0, 100, 1000) #创建一个包含30个点的余弦波信号
#wave = x + 1
#transformed = np.fft.fft(wave)  #使用fft函数对余弦波信号进行傅里叶变换。
## print (np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9))  #对变换后的结果应用ifft函数，应该可以近似地还原初始信号。
#plt.subplot(211)
#plt.plot(x,wave)
#plt.subplot(212)
#plt.plot(abs(transformed))  #使用Matplotlib绘制变换后的信号。
#plt.show()


def wave(max,size):
    x = np.arange(0,max,max/size)
    y = x+1
    return x,y
def freq_center(fy,size):
    #把频谱移到坐标原点
    fy_center = fy.copy()
    
    offset = int(size/2)
    for i in range(0,size,1):
        if i < offset:
            fy_center[i] = fy[i+offset]
        else:
            fy_center[i] = fy[offset-i]
        
    return fy_center
    
    
    
    
fft_size = 256
t = 10
x,y = wave(t,fft_size)
fy = np.fft.fft(y) / fft_size

f_center = np.arange(-1/(2*t),1/(2*t), 1/(t*fft_size))


fy_center = freq_center(fy,fft_size)


plt.subplot(211)
plt.plot(x,y)
plt.grid()
plt.subplot(212)
plt.plot(f_center,20*np.log10(np.abs(fy_center)))
plt.grid()
