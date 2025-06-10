import threading

class MessageCounter:
    """
    Thread-safe singleton class to manage a message counter.
    The purpose of this class is to generate consistently unique and sequential message IDs.
    The counter is initialized only once and can be accessed from multiple threads.
    The counter is incremented atomically to ensure thread safety.
    
    NOTE:
    - This class is not responsible for persisting the counter value.
    - The initial counter value should be stored in a persistent storage (e.g., database) and retrieved when initializing the counter.
    - This counter value should be updated in the persistent storage after each increment.
    
    Usage:
    >>> counter = MessageCounter()
    >>> counter.initialize(1000)
    >>> print(counter.get())  # Output: 1001
    >>> print(counter.get())  # Output: 1002    
    """
    
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._counter = 0
                    cls._instance._counter_lock = threading.Lock()
                    cls._instance._initialized = False
        return cls._instance

    def initialize(self, initial_value: int):
        """Thread-safe one-time initialization of the counter."""
        with self._counter_lock:
            self._counter = initial_value
            self._initialized = True

    def get(self) -> int:
        """Atomically increment and return the counter."""
        with self._counter_lock:
            self._counter += 1
            return self._counter
