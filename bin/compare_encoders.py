from joblib import Parallel, delayed
from sys import argv
from Bjontegaard import *
from subprocess import check_output, PIPE, STDOUT
from os import system
from os.path import isfile

def runEncoder(cmd, encoder1,encoder2):

	seqMap = {'RaceHorsesC':'RHC', 'BQSquare': 'BQS','BQMall':'BQM','BasketballDrill' : 'BDril','BasketballDrive':'BDrv','BQTerrace':'BQT',
			  'PartyScene':'PtS','ParkScene':'PkS','Cactus':'CAC','Kimono':'KIM', 'BasketballPass' : 'BPas'}

	re_bitrate = '%d\s*a\s*(\d+.\d+)\s*' % (num_frames)
	re_psnr = '%d\s*a\s*\d+.\d+\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*' % (num_frames)
	re_time = 'Total\sTime:\s*(\d+.\d+)\s*'
	cmd_split = cmd.split()
	qp = int(cmd_split[cmd_split.index('-q')+1])
	if encoder1 in cmd:
		encoder = encoder1
	else:
		encoder = encoder2
	
#	print 'Running: ', cmd

	cfg = 'RA' if 'randomaccess' in cmd else 'LB'
	seq = seqMap[cmd.split('~/hm-cfgs/')[1].split()[0][:-4]]
	qpstr = cmd.split(' -q ')[1].split()[0]
	fr = cmd.split(' -f ')[1].split()[0]
	extra_opt = cmd.split(' -f')[1].split()[1:]
	#print extra_opt
	if len(extra_opt) >= 1:
		opt_str = '_'.join([x.replace('--','').split('=')[0][:6] + x.split('=')[1] for x in extra_opt])
		cmd_name = '%s-%s-%s-%s-qp%s-%sfr.txt' % (encoder,seq, cfg,opt_str,qpstr,fr)
	else:
		cmd_name = '%s-%s-%s-qp%s-%sfr.txt' % (encoder,seq, cfg,qpstr,fr)

	outf_path = 'OUTPUT/%s' % (cmd_name)
	if not isfile(outf_path):
		print 'Running',outf_path
		output = check_output(cmd, shell=True,stderr=PIPE)
		fout = open(outf_path,'w')
		fout.write(output)
		fout.close()
	else:
		print 'Running',outf_path
		output = open(outf_path,'r').read()
	# ./hm-fast-fme-v1 -c ../cfg/encoder_randomaccess_main.cfg -c ~/hm-cfgs/BasketballDrill.cfg -q 37 -f 30


	bitrate = re.search(re_bitrate, output).group(1)
	time = re.search(re_time,output).group(1)
	[y,u,v,yuv] = re.search(re_psnr,output).groups()
	#print encoder,qp,time,bitrate, y,u,v,yuv
	return [encoder,qp,time,bitrate, y,u,v,yuv]

encoder1 = argv[1]
test_encoders = argv[2].split(',')

sequences = argv[3].split(',')
num_frames = int(argv[4])

qps = [22,27,32,37]
gop_struct = 'encoder_randomaccess_main'

N_CORES = 1

extra_opt1 = '--UseSVMCU=0'
extra_opt2 = [#'--FTU=0 --FFME=2 --FMEth1=0.5 --FMEth2=0.5',
			  #'--FTU=0 --FFME=2 --FMEth1=0.5805 --FMEth2=0.1007',
			  #'--FTU=0 --FFME=2 --FMEth1=0.4195 --FMEth2=0.5',
			  #'--FTU=2 --FFME=2 --FMEth1=0.5 --FMEth2=0.5',
			  #'--FTU=2 --FFME=2 --FMEth1=0.5805 --FMEth2=0.1007',
			  '--FTU=2 --FFME=0 --FMEth1=0.4195 --FMEth2=0.5', # TU Fast Only
			  '--FTU=0 --FFME=2 --FMEth1=0.4195 --FMEth2=0.5', # FME Fast Only
			  '--FTU=2 --FFME=2 --FMEth1=0.4195 --FMEth2=0.5',
			  '--FTU=2 --FFME=2 --FMEth1=0 --FMEth2=0 --TUth0=0 --TUth1=1 --TUth2=1',#NO TU
			  '--FTU=2 --FFME=2 --FMEth1=0 --FMEth2=1 --TUth0=0 --TUth1=0 --TUth2=0',#NO FME QUARTER
			  '--FTU=2 --FFME=2 --FMEth1=1 --FMEth2=1 --TUth0=0 --TUth1=0 --TUth2=0',#NO FME
			  '--FTU=2 --FFME=2 --FMEth1=1 --FMEth2=1 --TUth0=0 --TUth1=1 --TUth2=1',#NO TU+FME
			  ]

extra_opt2 = ['--SvmScale=0 --UseSVMCU=1 --SvmThCU0=0.5 --SvmThCU1=0.5 --SvmThCU2=0.5',
			  ]
system('mkdir -p OUTPUT')

for encoder2 in test_encoders:
	for seq in sequences:
		for extra_opt in extra_opt2:
			cmds = []
			for qp in qps:
				base_opt = '-c ../cfg/%s.cfg -c ~/hm-cfgs/%s.cfg -q %d -f %d' % (gop_struct, seq, qp, num_frames)
				cmd1 = './%s %s %s' % (encoder1, base_opt, extra_opt1)
				cmds.append(cmd1)

			for qp in qps:
				base_opt = '-c ../cfg/%s.cfg -c ~/hm-cfgs/%s.cfg -q %d -f %d' % (gop_struct, seq, qp, num_frames)
				cmd2 = './%s %s %s' % (encoder2, base_opt, extra_opt)
				cmds.append(cmd2)

			results = Parallel(n_jobs = N_CORES)( delayed (runEncoder)(cmd,encoder1,encoder2) for cmd in cmds )
			refBDResults = [[] for _ in qps]
			refTimes = [[] for _ in qps]
			testBDResults = [[] for _ in qps]
			avg_tr = 0.0
			for result in results:
				qp = result[1]
				idx = qps.index(qp)
				time,bitrate, y,u,v,yuv = result[2:]

				if result[0] == encoder1:
					refBDResults[idx] = [float(bitrate),float(y),float(u),float(v),float(yuv)]
					refTimes[idx] = float(time)
				else:
					testBDResults[idx] = [float(bitrate),float(y),float(u),float(v),float(yuv)]
					tr = float(time)/float(refTimes[idx])
					avg_tr += tr
						
			bdr = bdrate(refBDResults, testBDResults)		
			print '%s\t%d\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.2f\t%.2f' % (seq,num_frames,encoder2,extra_opt,bdr[0],bdr[1],bdr[2],bdr[3],  avg_tr/len(qps),1-(avg_tr/len(qps)))

