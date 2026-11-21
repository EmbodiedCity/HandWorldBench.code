"""Versioned, score-independent native transition sampling across three temporal strata."""
import math

def strata(first,last,count=3):
 if not isinstance(first,int) or not isinstance(last,int) or first<0 or last-first+1<count*3:raise ValueError('insufficient_native_frames')
 n=last-first+1;edges=[first+(k*n)//count for k in range(count+1)]
 return [(edges[k],edges[k+1]-1) for k in range(count)]
def select_peak(differences,first,last):
 # differences[i] compares source i-1 to i. Keep three frames inside the stratum.
 candidates=range(first+1,last)
 values=[(i,float(differences[i])) for i in candidates]
 if not values or any(not math.isfinite(v) or v<0 for _,v in values):raise ValueError('invalid_transition_scores')
 peak=max(values,key=lambda x:x[1])[0]
 return [peak-1,peak,peak+1],float(differences[peak])
