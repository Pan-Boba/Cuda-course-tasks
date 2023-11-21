# Build
To build solution run **build.sh**, upon compling new directory *"/build"* is created.

**Exe** is located in *"/build/Release"*

[`CUDA Toolkit 11.7`](https://developer.nvidia.com/cuda-11-7-0-download-archive?) is used in this project, tasks from http://ccfit.nsu.ru/arom/en_207.


# 1 task results
Task:
>Allocate GPU array arr of $10^8$ float elements and initialize it with the kernel as follows: $arr[i] = \sin((i \\% 360) \cdot \pi / 180)$. Copy array in CPU memory and count error as $err = \sum_i |\sin((i \\% 360) \cdot \pi/180) - arr[i]| / 10^8$. Investigate the dependence of the use of functions: *sin*, *sinf*, *__sinf*. Explain the result. Check the result for array of double data type.

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

Все результаты получены при запуске локально на видеокарте *NVIDIA GeForce RTX 3050* и сильно зависят от последовательности запусков (из-за оптимизации и кэширования). При независимом запуске различных версий свертки получены следующие результаты по времени (в мксек) при  `BLOCK_SIZE` = 16:


| Memory type | Global | Shared | Texture 1D |
| :---: | :---: | :---: | :---: |
| Time, usec | 168 | 160 | 287 |

Из таблицы видно, что Shared память работает несколько быстрее остальных, но для нее расходуется время для переноса данных в память блоков. Использование texture-памяти не приносит особых плодов, она, очевидно, должна быть медленее разделяемой памяти, так как является общей внутри grid-а, но в тоже время в литературе указвается, что ее скорость работы выше чем у глобальной.  При `BLOCK_SIZE` равным 16 также посчитано среднее время выполнения каждого из при последовательном запуске каждого из алгоритмов по 3 раза:

```
Convolution with Global memory: average elapsed time 165.667 microseconds

Convolution with Shared memory: average elapsed time 163.667 microseconds

Convolution with Texture memory: average elapsed time 234 microseconds
```
Ускорение расчета для texture-памяти можно попробовать объяснить кэшированием в Texture Unit-е.

Пример использования фильтра размытия на изображении: 
<figure>
  <img src="https://github.com/Pan-Boba/Cuda-course-tasks/assets/102728548/801f66a0-c44b-4590-860c-19f7578db79c;auto=format&amp;fit=crop&amp;w=1000&amp;q=80" alt="">
</figure>


# 3 task results
Task:
>Modify the previous program so as to use all GPUs available for the program. The program should determine the amount of available GPU and distribute the work on them.

Для данного задания производился запуск *main.cu* на удаленном сервере **gpuserv**, для чего были созданы *smain.sbatch* и *makefile*. Изображения для обработки используется из предыдущего задания, загрузка и запись с ним проводятся с помощью [`libpng`](https://www.libpng.org/pub/png/libpng.html), при этом `BLOCK_SIZE` установлен равным 16.

Результаты получены для случая **2х** параллельных GPU при последовательном запуске каждого из алгоритмов по 3 раза:
```
Convolution with Global memory: average elapsed time 46.6667 microseconds
Convolution with Global memory: average elapsed time 44.3333 microseconds

Convolution with Shared memory: average elapsed time 25.3333 microseconds
Convolution with Shared memory: average elapsed time 23.6667 microseconds

Convolution with Texture memory: average elapsed time 68.3333 microseconds
Convolution with Texture memory: average elapsed time 64 microseconds
```

При запуске второго задания на этом же сервере, получена следующая таблица, содержащая время выполнения свертки с округлением до *int*:

|Num of GPU| Memory type | Global | Shared | Texture 1D |
| :---: | :---: | :---: | :---: | :---: |
| 1 | Time, usec | 105 | 107 | 114 |
| 2 | Time, usec | 45 | 44 | 66 |

Таким образом, получено ускорение обработки в **2** раза, без учета времени на дополнительное копирование, которое для выбранного изображения составляет несколько мсек.
