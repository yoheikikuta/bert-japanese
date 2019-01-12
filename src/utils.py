def str_to_value(input_str):
    """
    Convert data type of value of dict to appropriate one.
    Assume there are only three types: str, int, float.
    """
    if input_str.isalpha():
        return input_str
    elif input_str.isdigit():
        return int(input_str)
    else:
        return float(input_str)
