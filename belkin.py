'''
  
	Shared library for the Belkin contest.

	Author: Brian Chow
	Email: brian.a.chow@gmail.com
	Date: 2013-07-07

'''

from scipy import linspace, io
from pylab import *
from cmath import phase
from math import *
from matplotlib import pyplot
from bisect import bisect_left
import datetime

class DataFile:
	'''
    
    Class for Belkin Data files
    
    Contains methods for diagnostic plots, processing of training features, and other functions
    
  '''
	def __init__(self, filename, training = True):
		### load/calculate temporary data ###
		data = io.loadmat(filename)
		buf = data['Buffer']
		L1_P = buf['LF1V'][0][0] * buf['LF1I'][0][0].conjugate()
		L2_P = buf['LF2V'][0][0] * buf['LF2I'][0][0].conjugate()
		L1_ComplexPower = L1_P.sum(axis=1)
		L2_ComplexPower = L2_P.sum(axis=1)
		### store required data ###
		self.filename = filename
		# L1/L2 information stored as
			# 0 voltage
			# 1 current
			# 2 complex power
			# 3 real power
			# 4 imag power
			# 5 app power
			# 6 power factor
		self.L1 = array([buf['LF1V'][0][0],
			buf['LF1I'][0][0],
			L1_ComplexPower,
			L1_ComplexPower.real,
			L1_ComplexPower.imag,
			abs(L1_ComplexPower),
			[cos(phase(L1_P[i,0])) for i in range(len(L1_P))]])
		self.L2 = array([buf['LF2V'][0][0],
			buf['LF2I'][0][0],
			L2_ComplexPower,
			L2_ComplexPower.real,
			L2_ComplexPower.imag,
			abs(L2_ComplexPower),
			[cos(phase(L2_P[i,0])) for i in range(len(L2_P))]])
		self.L1_TimeTicks = array([np.int64(i) for i in buf['TimeTicks1'][0][0]])
		self.L2_TimeTicks = array([np.int64(i) for i in buf['TimeTicks2'][0][0]])
		self.HF = buf['HF'][0][0]
		self.HF_TimeTicks = array([np.int64(i) for i in buf['TimeTicksHF'][0][0]])
		if training:
			self.taggingInfo = buf['TaggingInfo'][0][0]
			self.tag_len = len(self.taggingInfo)
			self.training = True
		else:
			self.training = False
	def diagnostic_plot(self):
		### Class method to plot data ###
		fig = figure(1, figsize = (8, 6))
		### Plot real power consumption ###
		ax1 = fig.add_subplot(411)
		ax1.plot(self.L1[0,:], self.L1[4,:], color='blue')
		# ax1.set_title('Real Power (W) and ON/OFF Device Category IDs')
		for i in range(self.tag_len):
			ax1.plot([self.taggingInfo[i,2][0][0], self.taggingInfo[i,2][0][0]], 
				[0,4000], color='green', linewidth=2)
			ax1.plot([self.taggingInfo[i,3][0][0], self.taggingInfo[i,3][0][0]], 
				[0,4000], color='red', linewidth=2)
			str1 = 'ON-%s' % self.taggingInfo[i,0][0][0]
			ax1.text(self.taggingInfo[i,2][0][0], 4000, str1, rotation = 90)
			str2 = 'OFF-%s' % self.taggingInfo[i,0][0][0]
			ax1.text(self.taggingInfo[i,3][0][0], 4000, str1, rotation = 90)
		### Plot Imaginary/Reactive power (VAR) ###
		ax2 = fig.add_subplot(412)
		ax2.plot(self.L1[0,:], self.L1[5,:])
		# ax2.set_title('Imaginary/Reactive power (VAR)')
		### Plot Power Factor ###
		ax3 = fig.add_subplot(413)
		ax3.plot(self.L1[0,:], self.L1[7,:])
		# ax3.set_title('Power Factor');
		# ax3.set_xlabel('Unix Timestamp');
		### Plot HF Noise ###
		ax4_extent = (min(self.HF_TimeTicks)[0], max(self.HF_TimeTicks)[0], 0, 4096)
		pyplot.set_cmap('Blues')
		ax4.imshow(self.HF[::10, ::50], extent = ax4_extent, interpolation = "nearest")
		for i in range(len(taggingInfo)):
			plot([self.taggingInfo[i,2][0][0], self.taggingInfo[i,2][0][0]], [0,4000], color='green', linewidth=2)
			plot([self.taggingInfo[i,3][0][0], self.taggingInfo[i,3][0][0]], [0,4000], color='red', linewidth=2)
			str1 = 'ON-%s' % self.taggingInfo[i,0][0][0]
			text(taggingInfo[i,2][0][0],4000, str1, rotation = 90)
			str2 = 'OFF-%s' % self.taggingInfo[i,0][0][0]
			text(self.taggingInfo[i,3][0][0],4000, str2, rotation = 90)
		# ax4.set_title('High Frequency Noise')
		# ax4.set_ylabel('Frequency KHz')
		show()
		close()
	def get_start_times(self):
		return self.taggingInfo[:,2]
	def get_end_times(self):
		return self.taggingInfo[:,3]
	def get_appliance_ids(self):
		return self.taggingInfo[:,0]
	def get_appliance_names(self):
		return self.taggingInfo[:,1]
	def get_time_range(self):
		max_times = [max(self.HF_TimeTicks), max(self.L1_TimeTicks), max(self.L2_TimeTicks)]
		min_times = [min(self.HF_TimeTicks), min(self.L1_TimeTicks), min(self.L2_TimeTicks)]
		return [min(min_times), max(max_times)]
	def process_training_features(self):
		### Class method to extract training features ###
		# self.features structure:
			# 0 house
			# 1 app id
			# 2 timestamp
			# 3 ON (1) / OFF (0) status
			# 4 features
				# 0 HF
				# 1 L1
				# 2 L2
				# 	for both L1 and L2
				# 	0 LFv
				# 	1 LFi
				# 	2 Complex P
				# 	3 Real P
				# 	4 Imag P
				# 	5 App P
				# 	6 Power Factor
		if self.training:
			self.features = []
			fuzzy = 15
			start_times = self.get_start_times()
			end_times = self.get_end_times()
			appliance_ids = self.get_appliance_ids()
			for i in range(self.tag_len):
				start_time = start_times[i][0][0] - fuzzy
				self.features.append([self.filename[2:4], appliance_ids[i][0][0], start_time, 1, self.get_feature(i * 2, start_time)])
				end_time = end_times[i][0][0] - fuzzy
				self.features.append([self.filename[2:4], appliance_ids[i][0][0], end_time, 0, self.get_feature(i * 2 + 1, end_time)])
		else:
			### will eventually need to update this to extract relevant information for testing files ###
			print 'Data file does not contain training information'
	def get_training_features(self):
		### features is a list of length len(self.taggingInfo), and within each element are the features: ###
			# 0  Filename
			# 1  Feature ID (feature number in file)
			# 2  appliance ID
			# 3  appliance name
			# 4  [start time, end time]
			# 5  HF[interval]
			# 6  L1 array
			# 7  L2 array
		if self.training:
			return self.features
		else:
			print 'Data file does not contain training information'
			pass
	def get_feature(self, id_num, timestamp, app = None, interval = 60):
		### Extracts interval sec of features from a starting time stamp ###
		# store lists of TimeTicks
		HF_TT_list = self.HF_TimeTicks.tolist()
		L1_TT_list = self.L1_TimeTicks.tolist()
		L2_TT_list = self.L2_TimeTicks.tolist()
		# get minimum TimeTicks for validation
		min_time = [min(self.HF_TimeTicks), min(self.L1_TimeTicks), min(self.L2_TimeTicks)]
		max_time = [max(self.HF_TimeTicks), max(self.L1_TimeTicks), max(self.L2_TimeTicks)]
		# validate starting timestamp
		if [timestamp >= i for i in min_time]:
			start_time = timestamp
		else:
			print "TimeStamp is less than minimum TimeStamps!"
			return None
		# validate ending timestamp
		if [timestamp + interval <= i for i in max_time]:
			end_time = timestamp + interval
		else:
			end_time = min(max_time)
		# initiate features
		if self.training:
			features = []
		else:
			features = [id_num, self.filename[2:4], app, [start_time, end_time]]
		# get HF indicies corresponding with start/stop times and add to features
		start_index = index_search(HF_TT_list, start_time)
		end_index = index_search(HF_TT_list, end_time)
		features += [self.HF[:,start_index:end_index]]
		# get L1/L2 indicies corresponding to start/stop times, add to features
		start_index_1 = index_search(L1_TT_list, start_time)
		end_index_1 = index_search(L1_TT_list, end_time)
		features += [self.L1[:,start_index_1:end_index_1]]
		start_index_2 = index_search(L2_TT_list, start_time)
		end_index_2 = index_search(L2_TT_list, end_time)
		features += [self.L2[:,start_index_2:end_index_2]]
		return features

class Feature():
  '''
  
    Class for features
    
    Currently doesn't do much.
    
  '''
	def __init__(self, args, training = True):
		self.filename, self.feature, self.app_id, self.app_name, self.time_interval, self.HF, self.L1, self.L2 = args
		if training:
			pass
	### will need class methods to process data for feature engineering ###
		# e.g. seasonality, time intervals (time of day, week, month, year)
	def diagnostic_plot(self):
		### Create diagnostic plots of the feature ###
			# currently plots:
			# 1 L1/L2 real power
			# 2 L1/L2 imaginary power
			# 3 L1/L2 power factor
			# 4 HF spectra\
		# declare misc info (colors, axes, etc)
		L_col = ['blue', 'red'] # colours for L1 and L2, respectively
		num_L1 = np.shape(self.L1)[1]
		num_L2 = np.shape(self.L2)[1]
		times = [self.time_interval[1] - self.time_interval[2] for i in range(num_L1)]
		L1_times = [times[i] / range(num_L1, 0, -1) for i in range(num_L1)]
		L2_times = [times[i] / range(num_L2, 0, -1) for i in range(num_L2)]
		# create figure container
		fig = figure(1, figsize = (8, 6))
		# set figure title
		str1 = self.filename[2:4] + " " + self.feature + ": " + self.app_id + str(self.app_name[0])
		fig.set_title(str1)
		# set colourmap for HF plot
		pyplot.set_cmap('afmhot')
		# plot 1) real power vs time
		ax1 = fig.add_subplot(411)
		ax1.plot(L1_times, self.L1[:,3], color = L_col[0])
		ax1.plot(L2_times, self.L2[:,3], color = L_col[1])
		ax1.set_title('Real Power (W) vs Time')
		# plot 2) Imaginary power vs time
		ax2 = fig.add_subplot(412)
		ax2.plot(L1_times, self.L1[:,4], color = L_col[0])
		ax2.plot(L2_times, self.L2[:,4], color = L_col[1])
		ax2.set_title('Imaginary Power (VA) vs Time')
		# plot 3) Power Factor vs time
		ax3 = fig.add_subplot(413)
		ax3.plot(L1_times, self.L1[:,6], color = L_col[0])
		ax3.plot(L2_times, self.L2[:,6], color = L_col[1])
		ax3.set_title('Power Factor vs Time')
		# plot 4) HF vs time
		ax4 = fig.add_subplot(414)
		ax4_extent = (self.time_interval[0], self.time_interval[1], 0, 4096)
		ax4.imshow(self.HF, extent = ax4_extent, interpolation = "nearest")

def unix_to_date(timestamp):
  ### converts unix time stamps to dates ###
	return datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

def logical_xor(str1, str2):
    return bool(str1) ^ bool(str2)

def hamming_loss(pred, y, num_app, num_lab):
  ### Hamming loss function for the competition ###
	if len(pred) == len(y):
		sum_xor = 0.
		for i in range(len(pred)):
			sum_xor += (logical_xor(pred[i], y[i]) / num_app)
		return (1 / num_lab) * sum_xor
	else:
		print "Length of predictions and truths do not match"
		pass

def index_search(t, x):
    i = bisect_left(t, x)
    if t[i] - x > 0.5:
        i-=1
    return i

def diagnostic_plot(feature, id_num = None, app_dict = tags_dict):
	'''
    Plots real power, imaginary power, power factor, and high-frequency spectrum given a feature array (extracted from DataFile.process_training_features)
  '''
  ### Extract feature data ###
	house, app_id, timestamp, status, feats = feature
	HF, L1, L2 = feats
	if house == 'H1':
		app_dict = app_dict[0]
	elif house == 'H2':
		app_dict = app_dict[1]
	elif house == 'H3':
		app_dict = app_dict[2]
	elif house == 'H4':
		app_dict = app_dict[3]
	if status == 1:
		statusstr = 'ON'
	else:
		statusstr = 'OFF'
	### Declare vars ###
	L_col = ['blue', 'red'] # colours for L1 and L2, respectively
	num_HF = np.shape(HF)[1]
	num_L1 = np.shape(L1)[1]
	num_L2 = np.shape(L2)[1]
	HF_times = [((60. / num_HF) * i) + timestamp for i in range(num_HF)]
	L1_times = [((60. / num_L1) * i) + timestamp for i in range(num_L1)]
	L2_times = [((60. / num_L2) * i) + timestamp for i in range(num_L2)]
	font = {'family' : 'normal',
		'weight' : 'bold',
		'size'   : 8}
	matplotlib.rc('font', **font)
	# create figure container
	fig = figure(1, figsize = (8, 6), dpi = 128, tight_layout = True)
	titlestr = house + ' ' + app_dict[app_id] + ' ' + statusstr
	fig.suptitle(titlestr, x = 0.15)
	# set figure title
	str1 = house + ": Appliance " + str(app_id)
	# set colourmap for HF plot
	pyplot.set_cmap('cool')
	# plot 1) real power vs time
	ax1 = fig.add_subplot(411)
	ax1.plot(L1_times, L1[3], color = L_col[0])
	ax1.plot(L2_times, L2[3], color = L_col[1])
	ax1.set_xlim([min(L1_times), max(L1_times)])
	ax1.set_title('Real Power (W) vs Time')
	# plot 2) Imaginary power vs time
	ax2 = fig.add_subplot(412)
	ax2.plot(L1_times, L1[4], color = L_col[0])
	ax2.plot(L2_times, L2[4], color = L_col[1])
	ax2.set_xlim([min(L1_times), max(L1_times)])
	ax2.set_title('Imaginary Power (VA) vs Time')
	# plot 3) Power Factor vs time
	ax3 = fig.add_subplot(413)
	ax3.plot(L1_times, L1[6], color = L_col[0])
	ax3.plot(L2_times, L2[6], color = L_col[1])
	ax3.set_xlim([min(L1_times), max(L1_times)])
	ax3.set_title('Power Factor vs Time')
	# plot 4) HF vs time
	ax4 = fig.add_subplot(414)
	ax4_extent = (min(HF_times), max(HF_times), 0, 4096)
	ax4.imshow(HF, extent = ax4_extent, interpolation = "nearest", aspect = 'auto')
	ax4.set_title('High Frequency Spectrum vs Time')
	savestr = './Diagnostic Figures/' + str(id_num) + '_' + house + '_' + str(app_id) + '.pdf'
	savefig(savestr)
	close()
