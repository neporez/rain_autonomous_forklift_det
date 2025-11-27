import time
import functools
import sys
from collections import OrderedDict, deque

class PerformanceCollector:
    def __init__(self, window_size=1000):
        self.current_cycle = OrderedDict()
        self.cycle_count = 0
        self.displayed_lines = 0
        
        # 슬라이딩 윈도우 방식으로 변경
        self.window_size = window_size
        self.statistics = {}  # {name: deque of measurements}
        self.averages = {}
    
    def record(self, name, ms):
        self.current_cycle[name] = ms
        
        # 각 측정값에 대해 deque 초기화 (최초 1회)
        if name not in self.statistics:
            self.statistics[name] = deque(maxlen=self.window_size)
        
        # 최근 N개만 유지 (deque의 maxlen이 자동 처리)
        self.statistics[name].append(ms)
        
        # 평균 계산 (항상 최신 상태 유지)
        self.averages[name] = sum(self.statistics[name]) / len(self.statistics[name])
    
    def display_cycle(self):
        if self.displayed_lines > 0:
            sys.stdout.write(f'\033[{self.displayed_lines}A\033[J')
        
        lines = []
        lines.append('=' * 80)
        lines.append(f'Frame #{self.cycle_count}')
        lines.append('-' * 80)
        
        # 평균값 표시 (데이터가 있으면 항상 표시)
        has_avg = len(self.averages) > 0
        
        if has_avg:
            # 윈도우 크기 표시
            sample_count = len(next(iter(self.statistics.values()))) if self.statistics else 0
            lines.append(f'{"Module":30s}  {"Current":>10s}  {"Avg(last {})".format(min(sample_count, self.window_size)):>20s}')
            lines.append('-' * 80)
        
        for name, ms in self.current_cycle.items():
            if has_avg and name in self.averages:
                sample_count = len(self.statistics[name])
                lines.append(f'  {name:30s}: {ms:7.2f} ms  (avg: {self.averages[name]:7.2f} ms, n={sample_count})')
            else:
                lines.append(f'  {name:30s}: {ms:7.2f} ms')
        
        lines.append('=' * 80)
        
        output = '\n'.join(lines)
        print(output)
        sys.stdout.flush()
        
        self.displayed_lines = len(lines)
        self.cycle_count += 1
    
    def reset_cycle(self):
        self.current_cycle.clear()

# 전역 collector (window_size를 원하는 값으로 설정)
_collector = PerformanceCollector(window_size=1000)

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
