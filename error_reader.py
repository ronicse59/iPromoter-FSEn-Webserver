def FASTA(f):
    order = []
    sequences = {}

    for line in f:
        if line.startswith('>'):
            name = line[1:].rstrip('\n')
            name = name.replace('_', ' ')
            order.append(name)
            sequences[name] = ''

        else:
            sequences[name] += line.rstrip('\n')

    #print "%d sequences found" % len(order)
    return order, sequences