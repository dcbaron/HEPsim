from random import random, gauss, choice



class thingClass1:
    pass
index = thingClass1() #data wrapper; will keep track of objects, id numbers.
index.last={}

class thingClass:
    def __str__(self):
        name = str(self.__class__).split('.')[1].split('lass')[0][:-1]
        name = name + " " + str(self.id)
        return name

    def copy(self):
        copy = self.__class__() # most have initialization reqs
        for key in self.__dict__:
            copy.__dict__[key]  = self.__dict__[key]
        return copy

    def __init__(self):
        try:
            index.last[self.__class__] += 1
            self.id = index.last[self.__class__]
        except KeyError:
            index.last[self.__class__]=self.id=1
        print "making", self
        

class HEPclass:
    def __str__(self):
        name = str(self.__class__).split('.')[1].split('lass')[0][:-1]
        name = name + " " + str(self.id)
        return name
    def __init__(self,house):
        self.needsPrimary=True
        self.owner=0
        self.house=house
        self.percentage=0
        try:
            index.last[self.__class__] += 1
            self.id = index.last[self.__class__]
        except KeyError:
            index.last[self.__class__]=self.id=1
        print "making HEP", self.id#db

class neighborhoodClass:
    pass

class houseClass(thingClass):
    pass


class bidClass:
    def __init__(self,bidder,value):
        self.bidder = bidder
        self.value = value
        
    def copy(self):
        return bidClass(self.bidder,self.value)

class investorClass:
    def __init__(self):
        self.values = {}
        try:
            index.last[self.__class__] += 1
            self.id = index.last[self.__class__]
        except KeyError:
            index.last[self.__class__]=self.id=1

    def __str__(self):
        name = str(self.__class__).split('.')[1].split('lass')[0][:-1]
        name = name + " " + str(self.id)
        return name
        
    def getValu(self,house):
        try:
            self.values[house]['val'] += gauss(0, 50000*(t - self.values[house]['time'])**0.5)
            self.values[house]['time'] = t
        except KeyError:
            self.values[house] = {'val':500000 + gauss(0,50000) + gauss(0, 50000*t**0.5),
                                  'time':t}
        return self.values[house]['val']
    

class primaryClass:
    def __init__(self,hep):
        self.hep=hep
        self.bids=set()
        self.oldWinner=None
        self.startTime = t



realizations = 1
duration = 10./365 #years
longTimeStep = 1/12.
shortTimeStep = 1/365.
step = longTimeStep

def runPrimaries(step):
    
    for hep in heps:
        #print hep.house#db
        if hep.needsPrimary:
            hep.needsPrimary=False
            primaries.add(primaryClass(hep))
    if len(primaries) >0:
        toClose = set()
        for prim in primaries:
            #print prim.hep.house#db
            
            #get bids
            prim.lowbid = bidClass(None,1)
            secondbid = prim.lowbid.copy()
            prim.Tie = False

            for investor in investors:
                bid = investor.getValu(prim.hep.house)
                bid = int(1000*10000./bid)/1000.

                if bid == 0.999:  #db
                    print "bid fubar!"               #db
                    if raw_input()=='q':#db
                        return 1#db

                
                #print bid #db
                bid = bidClass(investor, bid)
                
                prim.bids.add(bid.copy())
                if bid.value <= prim.lowbid.value:
                    secondbid = prim.lowbid.copy()
                    prim.lowbid = bid.copy()
                    if secondbid.value==prim.lowbid.value:
                        prim.tie = True
                    else: prim.tie = False
                elif bid.value <= secondbid.value:
                    secondbid.value=bid.value

            #print "lowbid", prim.lowbid.value #db
            #if prim.oldWinner!=None: print "oldbid", prim.oldWinner.value #db
            #print "--------------------------"#db
                    
            if prim.tie:
                winners = []
                for bid in prim.bids:
                    if bid.value == prim.lowbid.value:
                        winners.append(bid)
                prim.winner = choice(winners)
                if prim.winner.value == 0.999:  #db
                    print "Tie : Winner fubar!"               #db
                    if raw_input()=='q':#db
                        return 1#db
            else:
                prim.lowbid.value = secondbid.value - 0.001
                prim.winner = prim.lowbid
            

                if prim.winner.value == 0.999:  #db
                    print "No tie : Winner fubar!"               #db
                    if raw_input()=='q':#db
                        return 1#db
                    

            # determine whether to close
            if (prim.oldWinner!=None and prim.oldWinner.value >= prim.winner.value):

                #close auctions
                prim.hep.owner = prim.oldWinner.bidder
                #p = prim.oldWinner.value; print p#db
                prim.hep.percentage = prim.oldWinner.value
                toClose.add(prim)

                print "Sold, after "+\
                    str(int((t-prim.startTime)/shortTimeStep)+1)+\
                    " days at auction:"
                print prim.hep
                print "Buyer:"
                print prim.hep.owner
                print "HEP claim percentage:"
                print prim.hep.percentage

                if prim.hep.percentage == 0.999:  #db
                    print "HEP fubar!"               #db
                    raw_input()                     #db
                    
                print
                
            else:
                prim.oldWinner = prim.winner.copy()

                if prim.oldWinner.value == 0.999:  #db
                    print "OW fubar!"               #db
                    raw_input()                     #db
                    #ow = prim.oldWinner
                    #print "SNAFU"
                    #print 'Value:',ow.value,".  Bidder:",ow.bidder
                    #print

        for prim in toClose:
            primaries.remove(prim)
        
        if len(primaries) > 0:
            step = shortTimeStep

            print "--------------------------------"#db
    
    return step

            
            

            

for realization in range(realizations):
    t = 0
    
    houses = set(houseClass() for i in range(100))
    heps = set(HEPclass(house) for house in houses)
    investors = set(investorClass() for i in range(2))
    primaries = set()
    
    while t<=duration:
        #print 't',t #db
        #print
        if (t/longTimeStep).is_integer():
            step = longTimeStep

        step = runPrimaries(step)
        t += step
        #print "----------------------------------------"



