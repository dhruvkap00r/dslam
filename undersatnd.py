from multiprocessing import Process

def print_s(text):
  for i in range(100):
    print(text)

def hell(af):
  for i in range(200):
    print(i)
if __name__== "__main__":

  p1 = Process(target=print_s, args=("hello ",))
  p2 = Process(target=hell, args=("nigga ",))
  p1.start()
  p2.start()


  p1.join()
  p2.join()

