from concurrent.futures import ThreadPoolExecutor

pool =ThreadPoolExecutor(100)#创造一个线程
pool.submit(函数名，参数1，参数2)#申请调用一个线程