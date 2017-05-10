from get_data import get_data_list

class SequenceStockData(object):
    """docstring for SequenceStockData"""

    def __init__(self, data_list, start, end):
        self.data_open = []
        self.data_close = []
        self.label = []

        for i in range(start, end):
            self.data_open.append(data_list[i]["open"])
            self.data_close.append(data_list[i]["close"])
            temp = float(data_list[i]["close"]) - float(data_list[i]["open"])
            self.label.append(temp)
        self.batch_id = 0


    def next_batch(self, batch_size):
        if self.batch_id == len(self.data_open):
            self.batch_id = 0
        batch_data = (self.data_open[self.batch_id:min(
            self.batch_id + batch_size, len(self.data_open))])
        batch_label = (self.label[self.batch_id:min(
            self.batch_id + batch_size, len(self.label))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data_open))

        return batch_data, batch_label

if __name__ == '__main__':
	data_list = get_data_list('test.csv')
	train = SequenceStockData(data_list, 0, 100)
	batch1, label1 = train.next_batch(10)
	batch2, label2 = train.next_batch(5)

	print(batch1)
	print(label1)
	print(batch2)
	print(label2)