import datetime

file_name = 'test.csv'

def serialize_line(line):
    line = line.strip().strip('\ufeff').split(',')
    data = {
        'time': datetime.datetime.strptime(line[0], '%Y-%m-%d %H:%M'),
        'open': line[1],
        'max': line[2],
        'min': line[3],
        'close': line[4],
        'settlement': line[5],
        'turnover': line[6],
        'volume': line[7],
        'interest': line[8],
    }
    return data

def get_data_list(file_name):
	data_list = []
	with open(file_name, encoding="utf-8") as input_file:
		for line in input_file.readlines():
			data = serialize_line(line)
			data_list.append(data)
	return data_list

if __name__ == '__main__':
	data_list = get_data_list(file_name)

	print(data_list)