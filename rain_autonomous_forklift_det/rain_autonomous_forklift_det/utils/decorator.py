import time
import functools
import sys
from collections import OrderedDict

class PerformanceCollector:
    def __init__(self):
        self.current_cycle = OrderedDict()
        self.cycle_count = 0
        self.displayed_lines = 0
    
    def record(self, name, ms):
        self.current_cycle[name] = ms
    
    def display_cycle(self):
        if self.displayed_lines > 0:
            sys.stdout.write(f'\033[{self.displayed_lines}A\033[J')
        
        lines = []
        lines.append('=' * 50)
        lines.append(f'Frame #{self.cycle_count}')
        lines.append('-' * 50)
        
        for name, ms in self.current_cycle.items():
            lines.append(f'  {name:30s}: {ms:7.2f} ms')
        
        lines.append('=' * 50)
        
        output = '\n'.join(lines)
        print(output)
        sys.stdout.flush()
        
        self.displayed_lines = len(lines)
        self.cycle_count += 1
    
    def reset_cycle(self):
        self.current_cycle.clear()

# 전역 collector
_collector = PerformanceCollector()

def measure_time(name=None):
    def decorator(func):
        disp_name = name if name is not None else func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            t1 = time.time()
            _collector.record(disp_name, (t1 - t0) * 1000)
            return result

        return wrapper

    if callable(name):
        func = name
        name = None
        return decorator(func)

    return decorator

def results_bar(func):
    """전체 사이클의 시작과 끝을 표시"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _collector.reset_cycle()  # 새 사이클 시작
        
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        
        _collector.record('>>> TOTAL <<<', (t1 - t0) * 1000)
        _collector.display_cycle()  # 모든 측정값 한 번에 표시
        
        return result
    return wrapper