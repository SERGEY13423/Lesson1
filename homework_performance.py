"""
homework_performance.py
Выполнение Задания 3: Сравнение производительности CPU vs CUDA.
"""

import time
import torch


def measure_cpu(func, *args, repeats=5):
    """
    Измеряет среднее время выполнения функции на CPU в миллисекундах.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)  # переводим в миллисекунды
    return sum(times) / len(times)


def measure_gpu(func, *args, repeats=5):
    """
    Измеряет среднее время выполнения функции на GPU через torch.cuda.Event в мс.
    """
    if not torch.cuda.is_available():
        return None

    # Прогрев GPU (warm-up), чтобы исключить время инициализации контекста CUDA
    func(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeats):
        start_event.record()
        func(*args)
        end_event.record()

        # Ждем физического завершения операций на GPU
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))  # уже возвращает мс

    return sum(times) / len(times)


def run_benchmarks():
    """Проводит бенчмарк базовых операций на CPU и GPU."""
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CUDA недоступна (только CPU)"
    print(f"Используемое устройство GPU: {device_name}\n")

    # Возьмем размерность для наглядного сравнения (например, 2048 x 2048)
    size = (2048, 2048)
    print(f"Тестируемый размер матриц: {size[0]}x{size[1]}")

    # Создаем тензоры на CPU
    A_cpu = torch.randn(size, dtype=torch.float32)
    B_cpu = torch.randn(size, dtype=torch.float32)

    # Если CUDA доступна, отправляем копии тензоров в память GPU
    if has_cuda:
        A_gpu = A_cpu.to('cuda')
        B_gpu = B_cpu.to('cuda')

    # Набор операций для тестирования
    operations = [
        ("Матричное умножение", lambda a, b: torch.matmul(a, b)),
        ("Поэлементное сложение", lambda a, b: a + b),
        ("Поэлементное умножение", lambda a, b: a * b),
        ("Транспонирование", lambda a, b: a.t()),
        ("Вычисление суммы", lambda a, b: a.sum())
    ]

    print("\n" + "=" * 65)
    print(f"{'Операция':<26} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
    print("=" * 65)

    for op_name, op_func in operations:
        cpu_time = measure_cpu(op_func, A_cpu, B_cpu)

        if has_cuda:
            gpu_time = measure_gpu(op_func, A_gpu, B_gpu)
            speedup = f"{cpu_time / gpu_time:.1f}x" if gpu_time > 0 else "N/A"
            gpu_time_str = f"{gpu_time:.2f}"
        else:
            gpu_time_str = "N/A"
            speedup = "N/A"

        print(f"{op_name:<26} | {cpu_time:<10.2f} | {gpu_time_str:<10} | {speedup:<10}")

    print("=" * 65 + "\n")


def task_3_4_analysis():
    """
    3.4 Анализ результатов (ответы на теоретические вопросы задания).
    """
    analysis_text = """
--- 3.4 Аналитический отчет по производительности ---

1. Какие операции получают наибольшее ускорение на GPU?
   Наибольшее ускорение (в десятки и сотни раз) получают вычислительно-ёмкие операции, 
   в первую очередь — МАТРИЧНОЕ УМНОЖЕНИЕ (torch.matmul). У него кубическая сложность O(N^3), 
   и оно великолепно параллелится на тысячи ядер CUDA и Tensor Cores.

2. Почему некоторые операции могут быть медленнее на GPU?
   Простые операции (например, сложение маленьких векторов или взятие суммы небольшого числа) 
   требуют мало вычислений, но имеют накладные расходы: запуск CUDA-ядра (kernel launch overhead) 
   и ожидание синхронизации могут длиться дольше, чем быстрый прогон по кэшу процессора.

3. Как размер матриц влияет на ускорение?
   На малых матрицах (например, 10x10) CPU быстрее или наравне с GPU из-за накладных расходов. 
   Но с ростом размера матриц (1024, 2048, 4096) GPU полностью загружает все свои блоки 
   параллельных вычислений, и преимущество видеокарты становится колоссальным.

4. Что происходит при передаче данных между CPU и GPU?
   Передача данных через шину PCI Express (вызовы .to('cuda') или .to('cpu')) — это одна 
   из самых медленных операций. Если постоянно гонять тензоры туда-сюда между операциями, 
   вся выгода от быстрой видеокарты будет потеряна. В хорошем коде данные загружают на GPU 
   один раз, проводят все расчёты там, и возвращают на CPU только итоговый результат.
"""
    print(analysis_text)


if __name__ == "__main__":
    run_benchmarks()
    task_3_4_analysis()