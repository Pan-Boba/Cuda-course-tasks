# Build
To build solution run **build.sh**, upon compling new directory *"/build"* is created.

**Exe** is located in *"/build/Release"*

[`CUDA Toolkit 11.7`](https://developer.nvidia.com/cuda-11-7-0-download-archive?) is used in this project, tasks from http://ccfit.nsu.ru/arom/en_207.


# 1 task results
Task:
>Allocate GPU array arr of $10^8 float elements and initialize it with the kernel as follows: $arr[i] = \sin((i \\% 360) \cdot \pi / 180)$. Copy array in CPU memory and count error as $err = \sum_i |\sin((i \\% 360) \cdot \pi/180) - arr[i]| / 10^8$. Investigate the dependence of the use of functions: *sin*, *sinf*, *__sinf*. Explain the result. Check the result for array of double data type.

Инициализация $10^8$ элементов массива происходит параллельно на GPU, а вот вычисление итоговой double ошибки на CPU, что занимает довольно много времени.

Результаты получены при запуске локально на видеокарте *NVIDIA GeForce RTX 3050*:
```
Errors for float array (sinf, sin, __sinf): 4.67949e-8 (Time: 2.29336e+06 microseconds); 1.19868e-8 (Time: 5.49713e+06 microseconds); 1.30149e-7 (Time: 2.53817e+06 microseconds);

Errors for double array (sinf, sin, __sinf): 4.67949e-8 (Time: 2.86556e+06 microseconds); 9.31932e-18 (Time: 5.70418e+06 microseconds); 1.30149e-7 (Time: 2.87575e+06 microseconds);
```

Источники ошибок: 
1) конверсия double -> float (для float массива с double-функцией)
2) использование более быстрых, но менее точных функций *__sin*, *sinf* (float-функции)

Также можно отметить, что время инициализации массивов может меняться, так что тут надо собирать статистику и показывать среднее, но приведенные значения не противоречат здравому смыслу (float считается быстрее, да и при других запусках установлено тоже самое)


# 2 task results
Task:
>Implement a program for applying filters to your images. Possible filters: blur, edge detection, denoising. Implement three versions of the program, namely, using global, shared memory and texture. Compare the time.

Прежде запуска необходимо установить последнюю версию библиотеку [`OpenCv`](https://opencv.org/releases/) на диск C и выставить переменную окружения c именем `OpenCV_DIR` и значением `C:/opencv/build`, при возникновении проблем можно воспользоваться [этой инструкцией](https://habr.com/ru/articles/722918/).  Для свертки изображения "/Cuda-course-tasks/src/task2/ImageSample.png" используется ядро [**Gaussian blur 3×3**](https://en.wikipedia.org/wiki/Kernel_(image_processing)).

Итоговые изображения, полученные для 3-х версий светки сохраняются в *"/Cuda-course-tasks/build/src/task2"* под именами: **OutputGlobalMemory.png**; **OutputSharedMemory.png**; **OutputTextureMemory.png**.

Все результаты получены при запуске локально на видеокарте *NVIDIA GeForce RTX 3050* и сильно зависят от значения `BLOCK_SIZE`, а также последовательности запусков (из-за оптимизации и кэширования). При независимом запуске различных версий свертки получены следующие результаты по времени (в мксек) от значения `BLOCK_SIZE`:


|  BLOCK_SIZE| 16 | 32 | 64 |
| :---:   | :---: | :---: | :---: |
| Global Memory, usec | 301 | 283 | 290 |
| Shared Memory, usec | 301 | 283 | 290 |
| Texture 1D Memory, usec | 301 | 283 | 290 |

Из таблицы видно TO DO. При `BLOCK_SIZE` равным 64 также посчитано среднее время выполнения каждого из при последовательном запуске каждого из алгоритмов по 3 раза:

```
Convolution with global memory: elapsed time 69 microseconds

Convolution with shared memory: elapsed time 34 microseconds

Convolution with texture memory: TO DO
```

Пример использования фильтра размытия на изображении: 
<figure>
  <img src="https://github.com/Pan-Boba/Cuda-course-tasks/assets/102728548/801f66a0-c44b-4590-860c-19f7578db79c;auto=format&amp;fit=crop&amp;w=1000&amp;q=80" alt="">
</figure>
