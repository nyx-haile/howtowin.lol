import json

class Dragon:
	
	def __init__(self):
		self.locale = "en_GB"
		self.version = "14.14.1"
		self.item_path = f'../dragon/{self.version}/data/{self.locale}/item.json'
		with open(self.item_path, 'r') as f:
			self.json = json.load(f)
	
	def get_item(self, item_number):
		return None

if __name__ == "__main__":
	d = Dragon()
	print(d.json.keys())
	print(d.get_item(0))
	print(d.get_item(6672))
	
