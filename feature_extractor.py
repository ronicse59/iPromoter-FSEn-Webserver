import methods
import numpy as np
from fasta_reader import FASTA


def generator(input_file_name, feature_list):
    order, sequences = FASTA(input_file_name)
    print("-> Window feature set generating ...");
    newstr = []
    for s in order:
        p = sequences[s]
        p = p.upper()
        each_feature_vector = []
        # Feature 1
        # Frequencey count
        # --------------------------------------------------------------------------------------
        if (feature_list[1]):
            a, c, g, t = methods.frequency_count(p, 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)
            each_feature_vector.append(g + c)

        # Feature 2
        # mean, variance and standard-deviation
        # --------------------------------------------------------------------------------------
        if (feature_list[2]):
            a, c, g, t = methods.frequency_count(p, 'A', 'C', 'G', 'T')
            x = [a, c, g, t]
            each_feature_vector.append(np.mean(x))
            each_feature_vector.append(np.var(x))
            each_feature_vector.append(np.std(x))

        # Feature 3
        # G-C Skew
        # AT-GC ratio
        # --------------------------------------------------------------------------------------
        if (feature_list[3]):
            value = 0
            for s in p:
                if s == 'G':
                    value = 1
                elif s == 'C':
                    value = -1
                else:
                    value = 0
                each_feature_vector.append(value)

            a, c, g, t = methods.frequency_count(p, 'A', 'C', 'G', 'T')
            if (g + c) == 0:
                val = 1
            else:
                val = g + c
            at_gc_ratio = (a + t) / val
            each_feature_vector.append(at_gc_ratio)

        # Feature 4
        # K-mar frequency count
        # K = 2
        # --------------------------------------------------------------------------------------
        if (feature_list[4]):
            each_feature_vector = each_feature_vector + methods.two_mar_frequency_count(p)

        # Feature 5
        # K = 3
        # --------------------------------------------------------------------------------------
        if (feature_list[5]):
            each_feature_vector = each_feature_vector + methods.three_mar_frequency_count(p)

        # Feature 6
        # K = 4
        # --------------------------------------------------------------------------------------
        if (feature_list[6]):
            each_feature_vector = each_feature_vector + methods.four_mar_frequency_count(p)

        # Feature 7
        # K = 5
        # --------------------------------------------------------------------------------------
        if (feature_list[7]):
            each_feature_vector = each_feature_vector + methods.five_mar_frequency_count(p)

        # Feature 8
        # K = 6
        # --------------------------------------------------------------------------------------
        if (feature_list[8]):
            each_feature_vector = each_feature_vector + methods.six_mar_frequency_count(p)

        # Feature 5
        # 2-mar and K-gap count
        # --------------------------------------------------------------------------------------
        if (feature_list[9]):
            for i in range(1, 75):
                each_feature_vector = each_feature_vector + methods.two_mar_k_gap(p, i)

        # Feature 6
        # 3-mar and right K-gap
        # --------------------------------------------------------------------------------------
        if (feature_list[10]):
            for i in range(1, 75):
                each_feature_vector = each_feature_vector + methods.three_mar_right_k_gap(p, i)

        # Feature 7
        # 3-mar and left K-gap
        # --------------------------------------------------------------------------------------
        if (feature_list[11]):
            for i in range(1, 75):
                each_feature_vector = each_feature_vector + methods.three_mar_left_k_gap(p, i)

        # Feature 8
        # Pattern matching with minmum 3 matching is acceptable
        # --------------------------------------------------------------------------------------
        if (feature_list[12]):
            tata1 = "TATAAT"
            tataR = "TAATAT"
            tata2 = "TATAAA"
            tata2R = "AAATAT"
            threshold = 3
            for i in range(6):
                each_feature_vector.append(methods.string_matching(p, tata1, threshold))
                tata1 = tata1[1:] + tata1[:1]

                each_feature_vector.append(methods.string_matching(p, tataR, threshold))
                tataR = tataR[1:] + tataR[:1]

                each_feature_vector.append(methods.string_matching(p, tata2, threshold))
                tata2 = tata2[1:] + tata2[:1]

                each_feature_vector.append(methods.string_matching(p, tata2R, threshold))
                tata2R = tata2R[1:] + tata2R[:1]

            tata1 = "TTGACA"
            tataR = "ACAGTT"
            for i in range(6):
                each_feature_vector.append(methods.string_matching(p, tata1, threshold))
                tata1 = tata1[1:] + tata1[:1]

                each_feature_vector.append(methods.string_matching(p, tataR, threshold))
                tataR = tataR[1:] + tataR[:1]

            each_feature_vector.append(methods.string_matching(p, "AACGAT", threshold))

        # Feature 9
        # Position distance count of A,C,G,T
        # --------------------------------------------------------------------------------------
        if (feature_list[13]):
            each_feature_vector.append(methods.distance_count(p, 'A'))
            each_feature_vector.append(methods.distance_count(p, 'C'))
            each_feature_vector.append(methods.distance_count(p, 'G'))
            each_feature_vector.append(methods.distance_count(p, 'T'))

        # Feature 10
        # Dinucleotide Parameters Based on DNasel Digestion Data.
        # --------------------------------------------------------------------------------------
        if (feature_list[14]):
            each_feature_vector.append(methods.dinucleotide_value(p))

        # Numeric value for A=1,C=2,G=3,T=4
        # --------------------------------------------------------------------------------------
        if (feature_list[15]):
            string = p
            #each_feature_vector.append(methods.numerical_position(string))
            for i in string:
                each_feature_vector.append(methods.scl(i))

        if (feature_list[16]):
            a, c, g, t = methods.frequency_count(p[:8], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

            a, c, g, t = methods.frequency_count(p[8:16], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

            a, c, g, t = methods.frequency_count(p[16:24], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

            a, c, g, t = methods.frequency_count(p[24:32], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

            a, c, g, t = methods.frequency_count(p[32:40], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

            a, c, g, t = methods.frequency_count(p[40:48], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

            a, c, g, t = methods.frequency_count(p[48:56], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

            a, c, g, t = methods.frequency_count(p[56:64], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

            a, c, g, t = methods.frequency_count(p[64:72], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

            a, c, g, t = methods.frequency_count(p[72:], 'A', 'C', 'G', 'T')
            each_feature_vector.append(a)
            each_feature_vector.append(c)
            each_feature_vector.append(g)
            each_feature_vector.append(t)

        if (feature_list[17]):
            percentage = [0.2, 0.4, 0.6, 0.8, 1.0]
            nucletide = ['A', 'C', 'G', 'T']
            for nucl in nucletide:
                for j in percentage:
                    each_feature_vector.append(methods.frequency_index(p, nucl, j))

        newstr.append(each_feature_vector)
        print(len(newstr[0]))

    return order, sequences, newstr