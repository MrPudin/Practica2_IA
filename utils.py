def read_csv(file_name):
    table = []
    with open(file_name) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                continue
            parsed = []
            for entry in line.split(","):
                entry = _cast_to(entry)
                parsed.append(entry)
            table.append(parsed)
    return table


def split_observations_and_labels(table):
    data, labels = [], []
    for row in table:
        data.append(row[:-1])
        labels.append(row[-1])
    return data, labels


def _cast_to(value_str):
    """
    Given a value represented as a string, try to convert it
    to a more specific type (int, float) or fail back to string.
    """
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def read_sms(file_name):
    messages, labels = [], []
    with open(file_name) as f:
        for line in f:
            label, message = line.strip().split("\t", maxsplit=1)
            messages.append(message)
            labels.append(label)
    return messages, labels
