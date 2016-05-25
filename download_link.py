import urllib
import urllib2
import threading
import os, sys
import time
    
class CallbackThread(threading.Thread):

    def __init__(self, callback):
        threading.Thread.__init__(self)
        self.callback = callback
        self.argv = 0
        self._stop = threading.Event()
        self.cv = threading.Condition()

    def stop(self):
        print "called thread stop"
        self._stop.set()
        self.cv.acquire()
        self.cv.notify()
        self.cv.release()

    def stopped(self):
        return self._stop.isSet()

    def set_argv(self, argv):
        self.argv = argv

    def call_callback(self, argv=None):
        if argv != None:
            self.argv = argv
        self.cv.acquire()
        self.cv.notify()
        self.cv.release()

    def run(self):
        self.cv.acquire()

        while not self.stopped():
            self.callback(self.argv)
            self.cv.wait()
        self.cv.release()

class DownloadLink:

    def __init__(self, download_link, destination_path, callback):
        self.download_link = download_link
        self.destination_path = destination_path
        self.callback = callback
        self.callback_t = CallbackThread(self.callback)
        self.content_length = self.__download_size()

    def __download_size(self):
        res = urllib2.urlopen(self.download_link)
        return long(res.headers['content-length'])

    def completed_so_far(self):
        # if file does not exist
        if not os.path.exists(self.destination_path):
            return -1
        return os.stat(self.destination_path).st_size

    def start(self):
        try:
            self.callback_t.start()
            st = time.time()
            f = urllib2.urlopen(self.download_link)
            cur_size = 0
            chunk_size = None

            if self.content_length <= 1024:
                chunk_size = self.content_length
            else:
                chunk_size = 1024

            with open(self.destination_path, "wb") as code:
                while cur_size < self.content_length:
                    data = f.read(chunk_size)
                    self.callback_t.call_callback(argv=cur_size)
                    code.write(data)
                    cur_size += chunk_size
            self.callback_t.stop()
            et = time.time()
            print "time take to download file", (et-st)/1000.0,"seconds"
        except KeyboardInterrupt:
            print "Interrupted!, Task stopped"
            self.callback_t.stop()

def get_progress(s):
    print 'download completed so far', round(s/float(dl.content_length) * 100, 2)

download_link = "https://s3-us-west-2.amazonaws.com/arya-vision-data/madan/imagenet_5_class.zip"
destination_path = "img.zip"
dl = DownloadLink(download_link, destination_path, get_progress)
dl.start()
