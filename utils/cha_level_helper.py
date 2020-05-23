# convert character to output_sequence
# inspired by https://github.com/zzw922cn/Automatic_Speech_Recognition
def output_sequence(y_cha):
    """
    first sequence with batch
    """
    sequence = []
    start = 0
    sequence.append([])

    for i in range(len(y_cha[0])):
        if y_cha[0][i][0] == start:
            sequence[start].append(y_cha[1][i])
        else:
            start += 1
            sequence.append([])


    # only print first suquence of batch
    index = sequence[0]
    seq = []
    for idx in index:
        if idx == 0:
            seq.append(' ')
        elif idx == 27:
            seq.append("'")
        elif idx == 28:
            continue
        else:
            seq.append(chr(idx+96))

    seq = ''.join(seq)
    return seq

def int_sequence_to_text(int_sequence):
    """
    retrun the text accoring to the int_sequence
    only one data set without getting out from the batch
    """
    seq = []
    for idx in int_sequence:
        if idx == 0:
            seq.append(' ')
        elif idx == 27:
            seq.append("'")
        elif idx == 28:
            continue
        else:
            seq.append(chr(idx+96))

    seq = ''.join(seq)
    return seq

def int_sequence_to_text_test(int_sequence):
    """
    retrun the text accoring to the int_sequence
    only one data set without getting out from the batch
    """
    seq = []
    for idx in int_sequence:
        if idx == 1:
            seq.append(' ')
        elif idx == 28:
            seq.append("'")
        elif idx == 28:
            continue
        else:
            seq.append(chr(idx+96-1))

    seq = ''.join(seq)
    return seq

