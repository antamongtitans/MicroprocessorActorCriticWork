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
		
	def update_rectangle_overlaps(self):
		compare_list = self.rectangle_list
		for rectangle in self.rectangle_list:
			num_overlaps = 0
			for comp_rect in compare_list:
				if (rectangle.is_intersect(comp_rect, 10)):
					num_overlaps = num_overlaps + 1
			#Need to update the actual list not just the retangle copy
			indexTmp = self.rectangle_list.index(rectangle)
			self.rectangle_list[indexTmp].set_num_overlap(num_overlaps)
		
	def update_rectangle_list_xy(self, x_cord, y_cord, iter):
		self.rectangle_list[iter].set_lower_left(x_cord, y_cord)
		
	def update_rectangle_list_x_max_y_max(self, x_bounds, y_bounds, iter):
		self.rectangle_list[iter].set_bounds(x_bounds, y_bounds)
		
	def get_num_macros(self):
		return len(self.rectangle_list)
		
	def get_chip_score(self):
		rectangle_list = self.get_rectangle_list()
		total_score = 0
		for rectangle in rectangle_list:
			total_score = total_score + rectangle.calculate_score()
		return total_score	
	
	#Print a chip into a graphic
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
			ax.add_patch(p)
			#plot lines
			#get connections for item
			connect = item.get_connect()
			ax.set_axis_off()
		return ax
		

class Rectangle_struct:
	def __init__(self, name, min_x, max_x, min_y, max_y):
		self.name = name
		self.min_x = min_x
		self.max_x = max_x
		self.min_y = min_y
		self.max_y = max_y
		self.middle = (((min_x + max_x) / 2), ((min_y + max_y) / 2))
		self.height = max_y - min_y
		self.width = max_x - min_x
		self.lower_left = (min_x, min_y)
		self.connect = []
		self.num_overlap = 0
		self.score = 0
		self.bounds_x = 0
		self.bounds_y = 0

	#Does the rectangle strucutre intersect with others part
	def is_intersect(self, other, buffer):
		if (self.min_x < (other.max_x + buffer) and (self.max_x + buffer) > other.min_x
			and self.min_y < (other.max_y + buffer) and (self.max_y + buffer) > other.min_y):
				return True
		return False
	
	def get_midpoint_dist(self, other):
		distance = ((self.middle[0] - other.middle[0])**2 + (self.middle[1] - other.middle[1])**2)**0.5
		return distance
	
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
	
	def get_middle_x(self):
		return self.middle[0]

	def get_middle_y(self):
		return self.middle[1]
		
	def get_connect(self):
		return self.connect
		
	def get_num_overlap(self):
		return self.num_overlap
		
	def get_total_conn_dist(self):
		connectList = self.get_connect()
		distance = 0
		for connection in connectList:
			distance = distance + self.get_midpoint_dist(connection)
		return distance

	#Set Data structure values
	#Want to restructure to make macros that are x,y points and then height and width 
	#stored or at least just add that functionality
	def set_bounds(self, x_bound, y_bound):
		self.bounds_x = x_bound
		self.bounds_y = y_bound
		
	def set_num_overlap(self, overlaps):
		self.num_overlap = overlaps
	
	def set_connect(self, listToSet):
		self.connect = listToSet
		
	def set_lower_left(self, x, y):
		#When reseting lower left coordinate need to set 
		#the midpoints as well as the upper left hand corner
		height = self.get_height()
		width = self.get_width()
		self.min_x = x
		self.min_y = y
		self.max_x = (x + width)
		self.max_y = (y + height)
		self.middle = (((self.min_x + self.max_x) / 2), ((self.min_y + self.max_y) / 2))
		
	def calculate_score(self):
		#Set the points for out of bounds to million
		pts_bounds = 10000
		distance = self.get_total_conn_dist()
		num_overlaps = self.get_num_overlap()
		score = (distance / 100) +  num_overlaps * 1000
		#Calculate if rectangle is in the bounds of chip add points accordingly
		if (self.bounds_x < self.min_x):
			score = score + pts_bounds
		if (self.bounds_y < self.min_y):
			score = score + pts_bounds
		if (0 > self.min_x):
			score = score + pts_bounds
		if (0 > self.min_y):
			score = score + pts_bounds
		self.score = score
		return score
