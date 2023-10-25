# Build
To build solution run **build.sh**, upon compling new directory "/build" is created.

**Exe** is located in "/build/Release"

Tasks from http://ccfit.nsu.ru/arom/en_207.


# 1 task results
Инициализация массивов (1е8 элементов) происходит параллельно на GPU, а вот вычисление итоговой double ошибки (которая рассчитывается как err = sum(abs(sin((i % 360) * Pi / 180) - arr[i]))) на CPU, что занимает довольно много времени.

Результаты получены при запуске локально на видеокарте NVIDIA GeForce RTX 3050:

>Errors for float array (sinf, sin, __sinf): 4.67949 (Time: 2.29336e+06 microseconds); 1.19868 (Time: 5.49713e+06 microseconds); 13.0149 (Time: 2.53817e+06 microseconds);
>
>Errors for double array (sinf, sin, __sinf): 4.67949 (Time: 2.86556e+06 microseconds); 9.31932e-10 (Time: 5.70418e+06 microseconds); 13.0149 (Time: 2.87575e+06 microseconds);

Источники ошибок: 
1) конверсия double -> float (для float массива с double-функцией)
2) использование более быстрых, но менее точных функций *__sin*, *sinf* (float-функции)

Также можно отметить, что время инициализации массивов может меняться, так что тут надо собирать статистику и показывать среднее, но приведенные значения не противоречат здравому смыслу (float считается быстрее, да и при других запусках установлено тоже самое)

# 2 task results
Прежде запуска необходимо установить последнюю версию библиотеку [`OpenCv`](https://opencv.org/releases/) на диск C и выставить переменную окружения c именем `OpenCV_DIR` и значением `C:/opencv/build`, при возникновении проблем можно воспользоваться [этой инструкцией](https://habr.com/ru/articles/722918/).  Для свертки изображения "/Cuda-course-tasks/src/task2/ImageSample.png" используется ядро [**Gaussian blur 3×3**](https://en.wikipedia.org/wiki/Kernel_(image_processing)).

Результат обработки для 3х версий светки сохраняются в "/Cuda-course-tasks/build/src/task2" под именами: **OutputGlobalMemory.png**; **OutputSharedMemory.png**; **OutputTextureMemory.png**.

Результаты получены при запуске локально на видеокарте NVIDIA GeForce RTX 3050:

>Convolution with global memory: elapsed time 69 microseconds
>
>Convolution with shared memory: elapsed time 34 microseconds
>
>Convolution with texture memory: TO DO


![Results](https://github.com/Pan-Boba/Cuda-course-tasks/assets/102728548/801f66a0-c44b-4590-860c-19f7578db79c)
