from .constants import YELLOW, END

def warning(msg):
    print('WARNING: %s%s%s' % (YELLOW, msg, END))
