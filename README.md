# Build
To build solution run **build.sh**, upon compling new directory "/build" is created.

**Exe** is located in "/build/Release"

# 1 task results
Инициализация массивов (1е8 элементов) происходит параллельно на GPU, а вот вычисление итоговой double ошибки на CPU, что занимает довольно много времени.

Результаты получены при запуске локально на видеокарте NVIDIA GeForce RTX 3050:

*Errors for float array (sinf, sin, __sinf): 4.67949 (Time: 2.29336e+06 microseconds); 1.19868 (Time: 5.49713e+06 microseconds); 13.0149 (Time: 2.53817e+06 microseconds);*

*Errors for double array (sinf, sin, __sinf): 4.67949 (Time: 2.86556e+06 microseconds); 9.31932e-10 (Time: 5.70418e+06 microseconds); 13.0149 (Time: 2.87575e+06 microseconds);*

Источники ошибок: 
1) конверсия double -> float (для float массива с double-функцией)
2) использование более быстрых, но менее точных функций *__sin*, *sinf* (float-функции)

Также можно отметить, что время инициализации массивов может меняться, так что тут надо собирать статистику и показывать среднее, но приведенные значения не противоречат здравому смыслу (float считается быстрее, да и при других запусках установлено тоже самое)
