"""
Configuration for Triton acceleration.
"""

# Global flag to enable/disable Triton acceleration
TRITON_ENABLED = True  # Enable by default

# Hardware availability cache
_TRITON_HW_AVAILABLE = None

def check_triton_hardware():
    global _TRITON_HW_AVAILABLE
    if _TRITON_HW_AVAILABLE is None:
        try:
            import triton
            _TRITON_HW_AVAILABLE = True
        except ImportError:
            _TRITON_HW_AVAILABLE = False
    return _TRITON_HW_AVAILABLE

def set_triton_enabled(enabled: bool) -> None:
    """
    Enable or disable Triton acceleration globally.
    
    Args:
        enabled: If True, Triton kernels will be used when available.
                 If False, falls back to PyTorch implementations.
    """
    global TRITON_ENABLED
    if enabled:
        # Check if hardware supports it
        if check_triton_hardware():
            TRITON_ENABLED = True
        else:
            # Don't enable if hardware not available
            import warnings
            warnings.warn("Triton hardware/library not available. Ignoring enable request.")
            TRITON_ENABLED = False
    else:
        TRITON_ENABLED = False

def get_triton_status() -> dict:
    """
    Get current Triton acceleration status.
    
    Returns:
        Dictionary with Triton availability and configuration info.
    """
    hw_avail = check_triton_hardware()
    return {
        'triton_enabled': TRITON_ENABLED and hw_avail,
        'triton_hardware_available': hw_avail,
    }


