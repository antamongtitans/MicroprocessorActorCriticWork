import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator
import math

class Chip:
	def __init__(self, name, height, width, rectangle_list):
		self.name = name
		self.height = height
		self.width = width
		self.rectangle_list = rectangle_list
		
	def get_name(self):
		return self.name
		
	def get_height(self):
		return self.height
		
	def get_width(self):
		return self.width
		
	def get_rectangle_list(self):
		return self.rectangle_list
	
	def print_plot(self):
		#Setup the figure
		fig = plt.figure(self.name)
		ax = fig.add_axes([0,0,1,1])

		#Determine which is larger height or width and use it for making square axis
		if (self.width > self.height):
			image_buf = self.width * 0.1
			outerEdge = self.width
		else:
			image_buf = self.height * 0.1
			outerEdge = self.height

		#Change axis to view height and width plus buffer
		ax.set_xlim(0-image_buf, outerEdge+image_buf)
		ax.set_ylim(0-image_buf, outerEdge+image_buf)

		# axes coordinates are 0,0 is bottom left and 1,1 is upper right
		p = patches.Rectangle(
			(0, 0), self.width, self.height,
			fill=False, clip_on=False
			)

		ax.add_patch(p)
		
		for item in self.rectangle_list:
			minx = item.get_minx()
			miny = item.get_miny()
			maxx = item.get_maxx()
			maxy = item.get_maxy()
			width = maxx - minx
			height = maxy - miny
			p = patches.Rectangle((minx, miny), width, height, 
				fill=False, clip_on=False)
			#ax.text((minx+width/2), (miny+height/2), item.get_name())
			ax.add_patch(p)
			#plot lines
			#get connections for item
			connect = item.get_connect()
		ax.set_axis_off()
		return ax
			#plt.show()
		

class Rectangle_struct:
	def __init__(self, name, min_x, max_x, min_y, max_y):
		self.name = name
		self.min_x = min_x
		self.max_x = max_x
		self.min_y = min_y
		self.max_y = max_y
		self.middle = (((min_x + max_x) / 2), ((min_y + max_y) / 2))
		self.connect = []

	def is_intersect(self, other, buffer):
		if (self.min_x < (other.max_x + buffer) and (self.max_x + buffer) > other.min_x
			and self.min_y < (other.max_y + buffer) and (self.max_y + buffer) > other.min_y):
			#print("Intersection between: " + str(self.name) + " and " + str(other.name))
			return True
		return False
		
	def get_width(self):
		return (self.max_x - self.min_x)
		
	def get_height(self):
		return (self.max_y - self.min_y)
	
	def get_minx(self):
		return self.min_x
		
	def get_maxx(self):
		return self.max_x
		
	def get_miny(self):
		return self.min_y
		
	def get_maxy(self):
		return self.max_y
		
	def get_middle(self):
		return self.middle
		
	def get_name(self):
		return self.name
		
	def set_connect(self, listToSet):
		self.connect = listToSet
	
	def get_middle_x(self):
		return self.middle[0]

	def get_middle_y(self):
		return self.middle[1]
		
	def get_connect(self):
		return self.connect
