import argparse
from itertools import repeat
import math
from multiprocessing.pool import Pool
from multiprocessing.pool import ThreadPool
import os
import sys
import time
import requests
import socket


class ProgressBar(object):
	"""Simple progress bar.
	Output example: 
 		100.00% [2152/2152]
	"""

	def __init__(self, total=100, stream=sys.stderr):
		self.total = total
		self.stream = stream
		self.last_len = 0
		self.curr = 0

	def Increment(self):
		self.curr += 1
		self.PrintProgress(self.curr)

		if self.curr == self.total:
			print('')

	def PrintProgress(self, value):
		self.stream.write('\b' * self.last_len)
		pct = 100 * self.curr / float(self.total)
		out = '{:.2f}% [{}/{}]'.format(pct, value, self.total)
		self.last_len = len(out)
		self.stream.write(out)
		self.stream.flush()

def DownloadMapper(t):
	"""Downloads an URL.
 	Args:
		t: url
	Returns:
		A pair of URL and content.
	"""
 
	url, downloads_dir, timestamp_exactness = t
	return url, DownloadUrl(url, downloads_dir, timestamp_exactness)

def DownloadUrl(url, downloads_dir, timestamp_exactness, max_attempts=5, timeout=5): # Actually it was 5
	"""Downloads a URL.
	Args:
		url: The URL.
		timestamp_exactness
		max_attempts: Max attempts for downloading the URL.
		timeout: Connection timeout in seconds for each attempt.
	Returns:
		The HTML at the URL or None if the request failed.
	"""
	# Get file id
	fileid = url.replace("http://web.archive.org/web/", "")
	fileid = fileid.replace("http://", "")
	htmlfileid = fileid.replace("/", "-") + ".html"
	
	try:
		with open(downloads_dir+"/"+htmlfileid) as f:
			return f.read()
	except IOError:
		pass

	# Change download url depending on the timestamp_exactness
	url_data = url.strip().split("/")
	# Update timestamp
	url_data[4] = url_data[4][:timestamp_exactness]
	url = "/".join(url_data)
	
	attempts = 0
	while attempts < max_attempts:
		try:
			req = requests.get(url, allow_redirects=True, timeout=timeout)

			if req.status_code == requests.codes.ok:
				content = req.text.encode(req.encoding)
				with open(downloads_dir+"/"+htmlfileid, 'wb+') as f:
					f.write(content)
				return content
			elif (req.status_code in [301, 302, 404, 503] and attempts == max_attempts - 1):
				return None
		except requests.exceptions.ConnectionError:
			pass
		except requests.exceptions.ContentDecodingError:
			return None
		except requests.exceptions.ChunkedEncodingError:
			return None
		except requests.exceptions.Timeout:
			pass
		except socket.timeout:
			pass
		except requests.exceptions.TooManyRedirects:
			pass

		# Exponential back-off.
		time.sleep(math.pow(2, attempts))
		attempts += 1

	return None


def ReadUrls(filename):
	"""Reads a list of URLs.
	Args:
		filename: The filename containing the URLs.
	Returns:
		A list of URLs.
	"""

	with open(filename) as f:
		return [line.strip('\n') for line in f]

def WriteUrls(filename, urls):
	"""Writes a list of URLs to a file.
	Args:
		filename: The filename to the file where the URLs should be written.
		urls: The list of URLs to write.
	"""

	with open(filename, 'w') as f:
		f.writelines(url + '\n' for url in urls)

# global variables
downloads_dir = "./dataset/data/xsum-raw-downloads"
timestamp_exactness = 14

def DownloadMapper(urls):
	"""Downloads a URL.
	Args:
		zipped_inp: urls_left_todownload, repeat(downloads_dir), repeat(timestamp_exactness)
	Returns:
		A pair of URL and content.
	"""
	global downloads_dir
	global timestamp_exactness
	zipped_inp = zip([urls], repeat(downloads_dir), repeat(timestamp_exactness))
	zipped = list(set(zipped_inp))
	zipped = zipped[0]
	url, downloads_dir, timestamp_exactness = zipped
	# print url
	return url, DownloadUrl(url, downloads_dir, timestamp_exactness)

def DownloadMode(urls_file, missing_urls_file, downloads_dir, request_parallelism, timestamp_exactness):
	"""Downloads the URLs for the specified corpus.
	Args:
		request_parallelism: The number of concurrent download requests.
		timestamp_exactness: Time stamp exactness (year, month, date, hr, minute and second)
	"""
		
	print('Downloading URLs from the %s file:' % urls_file)
	urls_full = ReadUrls(urls_file)
	
	urls_valid_todownload = urls_full[:]
	missing_urls_filename = missing_urls_file
	if os.path.exists(missing_urls_filename):
		print('Only downloading missing URLs')
		urls_valid_todownload = list(set(urls_full).intersection(ReadUrls(missing_urls_filename)))

	collected_urls = []
	missing_urls = []
	urls_left_todownload = urls_valid_todownload[:]
	print('urls left to download:' + str(len(urls_left_todownload)))
	results = []
	
	while(len(urls_left_todownload) != 0):
		
		with Pool(5) as pool:
			results = pool.map(DownloadMapper, urls_left_todownload)
			progress_bar = ProgressBar(len(urls_left_todownload))

  		# for urls in urls_left_todownload:
		# 	ans = DownloadMapper(zip([urls], repeat(downloads_dir), repeat(timestamp_exactness)))
		# 	results.append(ans)
		# 	progress_bar = ProgressBar(len(urls_left_todownload))
		
		try:
			for url, story_html in results:
				if story_html:
					collected_urls.append(url)
				else:
					missing_urls.append(url)

			progress_bar.Increment()
		except KeyboardInterrupt:
			print('Interrupted by user')
			break
		except TypeError:
			print('TypeError (probably a robot.txt case)')

		# Reset the urls_left_todownload for the rest of not-tried urls
		print('Reset urls_left_todownload - (collected_urls, missing_urls)')
		urls_toignore = collected_urls[:] +  missing_urls[:]
		urls_left_todownload = list(set(urls_left_todownload) - set(urls_toignore))
		print('urls left to download: ' + str(len(urls_left_todownload)))

	# Write all the final missing urls
	missing_urls = []
	missing_urls.extend(set(urls_valid_todownload) - set(collected_urls))

	WriteUrls(missing_urls_file, missing_urls)

	if missing_urls:
		print ('%d URLs couldn\'t be downloaded, see %s.'
			% (len(missing_urls), missing_urls_file))
		print('Try and run the command again to download the missing URLs.')

def main():
	parser = argparse.ArgumentParser(description='Download BBC News Articles')
	parser.add_argument('--request_parallelism', type=int, default=200)
	parser.add_argument('--context_token_limit', type=int, default=2000)
	parser.add_argument('--timestamp_exactness', type=int, default=14)
	args = parser.parse_args()
		
	urls_file_to_download = "./dataset/data/url_data/web_arxive_urls.txt"
	missing_urls_file = "./dataset/data/url_data/web_arxive_urls_missing.txt" 
	downloads_dir = "./dataset/data/xsum-raw-downloads"
	try:
		print('Creating the download directory.')
		os.mkdir(downloads_dir)
	except WindowsError:
		print('Download directory already exists.')
	
	DownloadMode(urls_file_to_download, missing_urls_file, downloads_dir, args.request_parallelism, args.timestamp_exactness)

if __name__ == '__main__':
	main()