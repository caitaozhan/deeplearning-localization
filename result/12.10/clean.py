
def clean(file_in, file_out):
    with open(file_in, 'r') as f_in, open(file_out, 'w') as f_out:
        counter = 0
        for line in f_in:
            if counter % 5 == 3:
                pass
            else:
                f_out.write(line)
            counter += 1

file_in  = 'log-deepmtl-all-vary_numintru'
file_out = 'log-deepmtl-yolo_simple-vary_numintru'

clean(file_in, file_out)

file_in  = 'log-deepmtl-all-vary_sendensity'
file_out = 'log-deepmtl-yolo_simple-vary_sendensity'

clean(file_in, file_out)

