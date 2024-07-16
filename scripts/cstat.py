#!/usr/local/env python3


#A set of helper functions for dealing with in-game stats

#sort/unsort
#Reads champion stat dictionaries and returns a list with the correct ordering of stats.

#follows the below convention

sort_order = ['abilityHaste', 'abilityPower', 'armor', 'armorPen', 'armorPenPercent', 'attackDamage', 'attackSpeed', 'bonusArmorPenPercent', 'bonusMagicPenPercent', 'ccReduction', 'cooldownReduction', 'health', 'healthMax', 'healthRegen', 'lifesteal', 'magicPen', 'magicPenPercent', 'magicResist', 'movementSpeed', 'omnivamp', 'physicalVamp', 'power', 'powerMax', 'powerRegen', 'spellVamp']

def sort(d: dict) -> list:
	l = []
	for k in sort_order:
		l.append(d.get(k, None))	
	return l

def load(l: list) -> dict:
	d = {}
	for i, v in enumerate(l):
		d[sort_order[i]] = v
	return d

def dump(d: dict) -> str:
	return str(sort(d))

all_stats = ['championStats', 'currentGold', 'damageStats', 'goldPerSecond', 'jungleMinionsKilled', 'level', 'minionsKilled', 'participantId', 'position', 'timeEnemySpentControlled', 'totalGold', 'xp']

mvp_stats = ['currentGold', 'totalGold', 'xp']

test = {'abilityHaste': 0, 'abilityPower': 0, 'armor': 21, 'armorPen': 0, 'armorPenPercent': 0, 'attackDamage': 25, 'attackSpeed': 100, 'bonusArmorPenPercent': 0, 'bonusMagicPenPercent': 0, 'ccReduction': 0, 'cooldownReduction': 0, 'health': 604, 'healthMax': 604, 'healthRegen': 0, 'lifesteal': 0, 'magicPen': 0, 'magicPenPercent': 0, 'magicResist': 30, 'movementSpeed': 330, 'omnivamp': 0, 'physicalVamp': 0, 'power': 333, 'powerMax': 333, 'powerRegen': 0, 'spellVamp': 0}	
