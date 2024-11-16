import json

class Dragon:
    def __init__(self):
        self.locale = "en_GB"
        self.version = "14.14.1"
        self.tree_path = f'../db/tree'
        self.item_path = f'../dragon/{self.version}/data/{self.locale}/item.json'
        self.json_init = 0
        self.tree_init = 0

#    def init_json(self):
#        if self.json_init != 0:
#            return None
#        self.json_init = 1
#        with open(self.item_path, 'r') as f:
#            self.item_dict = json.load(f)
#
#    def init_tree(self):
#		if self.tree_init != 0:
#			return None
#		self.tree_init = 1
#		with open(self.tree_path, 'r') as f:
#			self.tree = json.load(f)
#		#open other jsons
#		
#	def get_item(self, item_number):
#		self.init_json()
#		return self.item_dict['data'][str(item_number)]
#	
#	def item_from(self, item_number):
#		return self.get_item(item_number).get('from', [])
#	def item_into(self, item_number):
#		return self.get_item(item_number).get('into', [])
#	
#	def build(self, new_item, item_set={}):
#		#get item from item_dict
#		self.init_json()
#		new_item = str(new_item)
#		item = self.item_dict['data'][new_item]
#		#check if any item in item_set in builds_from for new_item
#		#return item_set diff which is of the form
#			# {key: -1,
#			# key: 0}
#		#so that item update can do for key in item_diff.keys():
#				#item_set[key] += item_diff[key]
#		#before updating
#		print(item)
#	
#	def patch_update(self):
#		#rebuild tree
#		self.init_json()
#		tree = {}
#		#print(self.item_from(6672))
#		for item in self.item_dict['data'].keys():
#			tree[item] = {}
#			for sub_item in self.item_from(item):
#				tree[item][sub_item] = tree[item].get(sub_item, 0) + 1
#				if tree[sub_item].keys() == self.item_from(sub_item):
#					print(tree[sub_item])
#			
#		with open(self.tree_path, 'w') as f:
#			json.dump(tree, f)
#
#		self.init_tree()
#
#	def build_sr_items(self):
#		self.SR = [item for item in d.item_dict['data'].keys() if d.get_item(item)['maps']['11'] == True]
#	
#	def item_update(self, item):
		#fetch items from item dict
		#recurse
		#return None
if __name__ == "__main__":
	d = Dragon()
	d.init_json()
	print(d.get_item(228008))
	print([item for item in d.item_dict['data'].keys() if d.get_item(item)['maps']['11'] == True])
	d.build_sr_items()
	print(d.SR.sort())
	#d.patch_update()
	#print(d.tree['6672'])
	#print(d.get_item(6672))
	#print(d.tree['1036'])
