##########################################################
##########################################################
###
### Welcome, one and all, to the world's first and only
###
###              Exceptionally Amazing
###           Home-Equity-Position-Market
###            Prognosticating Simulator!
###
### Brought to you by Daniel Baron and Emily Searle-White.
###
##########################################################
##########################################################


############################################
## Simulation settings
############################################
realizations = 1

# All time quantities are in years
duration = 5#10./365
longTimeStep = 1/12.    #distinct time-step sizes improve effiency
shortTimeStep = 1/365.
step = longTimeStep

agingRate = 0.02    # annual depreciation, in the absence of repairs.

sim = None
HomeSalesOn=False

#Neighborhood details:
def getSD(meanGeo, multiplier = 0.7):
    return (1-1/meanGeo)*multiplier + 1

nbhdDict = dict(city = 'St Louis',
          meanGeo = 1.063,
          medianPrice= 2.257 * 10**5,
          percent99= 3,
          pHIP= .52,
          pDisaster= 0,
          spawnRate = 3,
          initialHomes = 1
          )
nbhdDict['sdGeo']=getSD(nbhdDict['meanGeo'])
nbhdDicts = set(nbhdDict)


############################################
## Imports, Constants, and Utilities
############################################
from random import random, gauss, choice, \
     shuffle, lognormvariate, expovariate
from math import floor, log, factorial, exp
from types import InstanceType
gauss99 = 2.326347874
sellDateLambda = 1./13 * log(2)

class dataClass: last ={}
# This is just a data wrapper; keeps track of how many
# of each object have been made so far

index = dataClass()

class riskFree: rate = Rate = log(1.03)

def poissonvariate(lambd):
# Returns a Poisson-distributed random number.
    k = 0.
    p = random()
    
    # P(X = 0), P(X <=0)
    P= exp(-lambd)
    cumP = P

    while p>cumP:
        k+=1.
        P *= lambd/k    #P(X=k)
        cumP += P       #P(X<=k)
    return k

def mean_and_variance(iterable, corrected=False):
# Compute and return the mean and variance of a list
# of numbers.
    sumX = 0
    squareSumX = 0
    for i in iterable:
        sumX += i
        squareSumX += i**2

    try: n= float(len(iterable))
    except TypeError:
        n=0.
        for i in iterable:
            n += 1.
    mean = sumX/n
    
    if corrected: divisor = n-1.
    else: divisor = n
    variance = (squareSumX - n*mean**2)/divisor

    return {'mean':mean,'variance':variance}

        
############################################
## Class Declarations
############################################
## Much of the implementation relies on classes and instances
## to represent the abstract "HEPs", "Investors", etc.
## Some of these are little more than data wrappers, but
## there are also big chunks of the simulation procedure coded 
## here as class methods.

class thingClass:
# The parent class; everybody else inherits these attributes
    
    def __str__(self):
    # Gives each Thing a name for it to print, eg "HEP 28".
    # To do: make it easy to print all relevant data,
    # e.g. "HEP 28-- House: 9, Claim: 2.1%, Owner: Bob."
        name = str(self.__class__).split('.')[1].split('lass')[0][:-1]
        name = name + " " + str(self.id)
        return name

    def copy(self):
    # Makes a shallow copy 
        copy = self.__class__(**self.__dict__) # most have initialization reqs
        copy.__dict__.update(self.__dict__)
        return copy

    def __init__(self, **args):
    # Track how many Things of this class have been made,
    # give this new Thing an ID number.
        try:
            index.last[self.__class__] += 1
            self.id = index.last[self.__class__]
        except KeyError:
            index.last[self.__class__]=self.id=1

class HEPclass(thingClass):
# The Home Equity Positions.

    def __init__(self,house,**args):
        # Each new-minted HEP has a house, but,
        # as yet, no owner, nor fixed percentage.
        thingClass.__init__(self) 
        self.needsPrimary=True
        self.owner=None
        self.house=house
        self.percentage=args.get(   # Initialize at 1,
            'percentage',           # except for testing
            args.get('p',1))        # purposes.
        house.heps.add(self)

    def getReservePrice(self):
    # The minimum price that a seller will accept at auction.
        if self.owner == None:
            return gauss(.9, .1)*valu(self.house)
        else:
            return gauss(.9,.1)*self.owner.getValu(self.house)

    def expire(self):
    # When the underlying house is sold, remove this HEP from
    # the list and pay the investor.
        heps.remove(self)
        self.owner.liquidWealth += payout(self)
        self.owner.heps.remove(self)

    def transfer(self, newOwner, price=0):
    # Transfer the HEP to a new owner, usually as a result
    # of sale at auction.
        
        if self.owner != None:
            self.owner.heps.remove(self)
            self.owner.liquidWealth += price

        newOwner.heps.add(self)
        newOwner.liquidWealth  -= price
        self.owner = newOwner
        
        

hepClass = HEPclass

class neighborhoodClass(thingClass):
# Neighborhoods.

    def __init__(self,
                 #city,
                 #meanGeo,
                 #sdGeo,
                 #medianPrice,
                 #percent99,
                 #pHIP,
                 #pDisaster,
                 #spawnRate,
                 **args):
    # Parameters:
    #   1. meanGeo = e^(mu) is the geometric mean of annual 
    #   appreciation, usually something like 1.063.
    #   2. sdGeo = e^(sigma) is the geometric standard deviation .
    #   3. medianPrice is the median home sale price at t=0.
    #   4. percent99 is the 99th percentile of home prices here,
    #   divided by the median price. E.g. if 99% of homes cost less
    #   than double the median, then percent99 = 2.
    #       (If info on the top decile, half decile, whatever, is more
    #       readily available, this can be adjusted.)
    #   5. pHIP is the proportion of homeowners who reported at least
    #   one home improvement project in the last two years
    #   6. pDisaster is the proportion who reported at least one disaster
    #   repair project in the last two years.
    #   7. spawnRate determines how often new houses in this hood want HEPS.
        thingClass.__init__(self)
        self.__dict__.update(args)
        
        self.mu=log(self.meanGeo)
        self.sigma=log(self.sdGeo)
        self.appreciation={t:1}
        self.lastTime = t
        self.lambda_HIP = -log(1-self.pHIP)*0.5
        self.lambda_Disaster = -log(1-self.pDisaster)*0.5

        # Keep track of the houses in this hood with live HEPs:
        self.houses = set()

        self.priceMu = log(self.medianPrice)
        self.priceSigma = log(self.percent99)/gauss99
    
    def copy(self, t, tabulaRasa = True):
        copy = thingClass.copy(self)
        if tabulaRasa:
            copy.houses=set()
            copy.appreciation={t:1}
        return copy
            

    def appreciate(self, t):
    # app(t) = app(s)*exp( mu*(t-s) + sigma*sqrt(t-s)*X ),
    # where X~N(0,1).

        # Have we gone back in time?
        if t< self.lastTime:
            
            #Scrub all traces of the alternate timeline.
            future = True
            time=self.appreciation.keys()
            time.sort()
            time.reverse()
            for Time in time:
                if Time > t : self.appreciation.pop(Time)
                elif future:
                    future = False
                    self.lastTime = Time
                    break                

        dt = t-self.lastTime
        dApp = lognormvariate(self.mu * dt,self.sigma * dt**.5)
        self.appreciation[t] =  dApp * \
                               self.appreciation[self.lastTime]

        #update all houses in this hood:
        for house in self.houses:
            house.updateValue(dApp, dt)

        self.lastTime = t
        
        #either update all houses whenever neighborhood is updated,
        # or update everybody only once per month,
        # or keep dictionary of appreciation values.

    def getInitialPrice(self):
    # I assume here that home prices are log-normally distributed;
    # thus the median price determines the mean, mu, of the log-prices,
    # and sigma is such that P( X < percent99*median) = 0.99:
    #
    #       self.priceMu = log(self.medianPrice),
    #       self.priceSigma = log(self.percent99)/gauss99. 
        return lognormvariate(              \
            self.priceMu, self.priceSigma)  \
            * self.appreciation[self.lastTime]

    def getInitialMortgage(self):
    # The median current mortgage debt in the US, as a percentage of the
    # value of the home, is 71%; a 100% mortgage is roughly the 80th
    # percentile value. Homeowners seeking HEPs will, in general, have better
    # than average credit, but will also often be assuming a new mortgage
    # on a newly purchased home -- a mortgage not yet payed down.
    #
    # I assume here that these two effects roughly cancel out, so that
    # the cumulative distribution of debt/value ratios is the same in
    # the HEP world as in the world at large; I further assume that the
    # ratios are log-normally distributed.
    #
        log_point71 = -0.342490309      # log(0.71)
        mortgage_sigma = 0.406941146    # -log(0.71)/z80
        return lognormvariate(log_point71, mortgage_sigma)

    def newHouses(self, t):
    # How many houses need new HEPs created?
    # Poisson distributed.
        return poissonvariate(  
            self.spawnRate * (self.lastTime - t)
            )

        
class houseClass(thingClass):
# Houses. Most of the old neighborhood model is reproduced as methods
# of this class.
#
# I could, at present, move a lot of these methods to the neighborhood
# class, since all relevant parameters come from the hood. But I want
# eventually to include "homeowner profiles", e.g. credit history,
# first time homeowner status, moving habits, and it will be useful
# to have the methods here.

    def __init__(self, neighborhood, distribution=False):
        self.hood = neighborhood
        neighborhood.houses.add(self)
        self.value = neighborhood.getInitialPrice()
        self.M0 = neighborhood.getInitialMortgage()*self.value

        # Keep track of all HEPs derived from this house, and
        # the sum of their claim percentages.
        self.heps = set()
        self.percentageSum = 0

        # Get the date at which the homeowner will sell the
        # home, and track whether investors know the homeowner
        # is planning to sell.
        self.sellDate = self.getSellDate()
        self.knownSell = False
        
        # Approximate the cumulative probability distribution
        # of HEP values in this neighborhood.
        # We usually skip this step.
        if distribution:
            self.dist = HEP_distribution(self)

    def updateValue(self, dApp, dt):
    # The existing value appreciates by dApp,
    # depreciates because of aging.
    # Any discrete changes are then applied.
                
        self.value *= dApp*(1-agingRate)**(dt)
        
        n = self.getNumDiscrete(dt)
        v = self.getValDiscrete(dt, n)
        self.value += v

        #print 'HIPs: ',n
        #print 'HIP vals: ', v
        

    def getNumDiscrete(self,dt):
    # The AHS data includes the number of homeowners who report
    # having had home improvement projects or disaster repairs
    # in the last two years. If p is the proportion of such households,
    # N is the number of HIPs in a household in two years, and we assume
    # that N~Poisson, then
    #       P( N >= 1 ) = p   =>   P( N = 0 )= 1-p   =>   lambda = -log(1-p),
    # where lambda is the Poisson parameter.
        return poissonvariate(dt*self.hood.lambda_HIP)

    def getValDiscrete(self,dt,n):
    # If the number of discrete changes is given by the Poisson r.v.,
    # then their expected value is determined by the expectation of
    # neighborhood appreciation: since each neighborhood is a bunch
    # of houses, we must have
    #       E(House Appreciation) = E(Hood App.) = e^(mu+0.5*sigma^2).
    #
    # We then assume that the values are normally distributed.
    #
    # Eventually I will add disasters as a second type of discrete change.
    
        mean = self.value * \
               agingRate/((1-agingRate)*self.hood.lambda_HIP)
        sd = mean

        # The sum of n RVs, each N(m,sd), is N(n*m, sqrt(n)*sd).
        mean *=  n
        sd   *=  n**.5
        return gauss(mean, sd)

    def getSellDate(self):
    # Data from the NAHB indicates that median tenure of a homeowner
    # is about 13 years. The probability of moving in a given year is
    # not constant -- it's something like P(n) = 1+ 4/n -- but I am going
    # to model it as constant, p = 1 - 2^(-1/13), because doing so makes
    # this a Poisson process with lambda = -log(1-p) = 1/13 * log(2).
    #
    # This will make it easier to estimate the expected payoff of a HEP,
    # since the time until a homeowner sells will be exponentially distributed,
    # and the pdf of the expo distribution is really easy to integrate.
        if HomeSalesOn:
            return t + expovariate(sellDateLambda)

        else:
            return duration + 99

    def remove(self):
    # When the homeowner sells the home, all HEPs derived from it expire
    # and we take it out of all the relevant data objects.
        self.hood.houses.remove(self)
        houses.remove(self)
        
        for hep in self.heps:
            hep.expire()
            
        

class bidClass(thingClass):
# A bid: it has a bidder, and a value.
# The value can be in percentage points
# (if the bid is for the primary auction)
# or in dollars/whatevers
# (in the secondary market).
    def __init__(self,bidder,value,**args):
        self.bidder = bidder
        self.value = value
        
class investorClass(thingClass):
#Investors

    def __init__(self,**args):
        self.values = {}
        thingClass.__init__(self)
        
    def getValu(self,house):
    # An investor's "valuation" for a house
    # is the most he/she would pay for a hypothetical
    # 100% HEP claim on that house;
    # thus [ getValu(house) * p% ] is the investor's
    # valuation of a p% HEP claim on that house.

    # At present, this is just a random walk:
        try:
            self.values[house]['val'] += gauss(0, 50000*(t - self.values[house]['time'])**0.5)
            self.values[house]['time'] = t
        except KeyError:
            self.values[house] = {'val':500000 + gauss(0,50000) + gauss(0, 50000*t**0.5),
                                  'time':t}
        return self.values[house]['val']

    def getSellList(self):

    
class auctionClass(thingClass):
# The parent class for both primary and secondary auctions

    def __init__(self,hep,**args):
    # Each auction is for one HEP;
    # we track the day's bids,
    # the previous day's winning bid,
    # and the time elapsed since opening the auction.
    #
    # We also record the seller's reserve price.
        thingClass.__init__(self)
        self.hep=hep
        self.bids=set()
        self.startTime = t
        
        self.reserve = self.getBidValue(
            self.hep.getReservePrice()  )
        self.oldWinner=bidClass(None, self.reserve)

    def roundIt(self, x):
    # Round a valuation to the nearest discrete bid step.
    # Since primaryClass.discrete is negative, this
    # rounds *up*: the investor won't bid any lower.
    #
    # For secondaries it of course rounds down.

        return floor(float(x)/self.discrete)*self.discrete

    def close(self,price):
    # When the HEP is sold, close the auction, transfer
    # ownership. If the reserve price wasn't met, then
    # of course no transfer occurs.

        winner = self.oldWinner.bidder
        if winner != None:
            hep = self.hep
            hep.transfer(winner, price)

        auctions.remove(self)

        print "AUCTION REPORT: ", self
        if winner == None:
            print self.hep
            print "Reserve not met."
        else:
            print "Sold, after "+\
                str(int((t-self.startTime)/shortTimeStep)+1)+\
                " days at auction:"
            print self.hep
            print "Buyer:"
            print winner
            if self.__class__ == primaryClass:
                print "HEP claim percentage:"
                print self.oldWinner.value
            else:
                print "Sale price"
                print self.oldWinner.value
                print "HEP claim percentage:"
                print hep.percentage
        print

    def 
        

class primaryClass(auctionClass):
# One primary auction.

    def initializeWinner(self):
    # We need to ensure that the sum of the claim percentages
    # of all HEPs on a house is never greater than 50%. 
        Sum = self.hep.house.percentageSum        
        if self.oldWinner.value <= 0.5 - Sum:
            return self.oldWinner
        else:
            value = 0.5 - Sum
            bidder = None
            return bidClass( bidder, value )

    def isAsGood(self,bid1, bid2):
    # Is bid1 at least as good as bid2?
    # *Lowest* bid wins.
        return bid1.value <= bid2.value
    
    def close(self):
    # When the HEP is sold, close the auction,
    # transfer ownership,
    # fix the HEP claim percentage.

    # If reserve price wasn't met, the HEP is never created:
        if winner == None:
            self.hep.delete()
            
        else:
            self.hep.percentage = self.oldWinner.value
            self.hep.house.percentageSum += \
                                         self.hep.percentage

        price = 10000
        auctionClass.close(self, price)
        
    def getBidValue(self, valu):
    # The price of a new HEP is fixed at $10,000;
    #   10,000 / valu  =  p% / 100%.
        return self.roundIt(10000./valu)

    discrete = - 0.001      #percentage points
    # The bids come in discrete steps.
    # In addition to being accurate (auctions don't
    # work if I can raise the bid by 1/10 cent
    # to win it from you), this makes for the interesting
    # possibility of tie bids.
    #
    # This is negative because we want low bids.

class secondaryClass(auctionClass):
# The class of secondary auctions.

    def initializeWinner(self):
        return self.oldWinner

    def isAsGood(self,bid1, bid2):
    # Is bid1 at least as good as bid2?
    # *Highest* bid wins.
        return bid1.value >= bid2.value

    def close(self):
    # When the HEP is sold, close the auction,
    # transfer ownership.
        price = self.oldWinner.value
        auctionClass.close(self, price)

    def getBidValue(self, valu):
    # valu == what I would pay for a 100% HEP.
    # Thus, getBidValue == p% * valu == what I 
    # pay for a p% HEP.
        return self.roundIt(self.hep.percentage * valu)

    discrete = 100 #dollars


#####################################################
#   the Auction procedures.
#####################################################

# Regarding "bids":
#
# A "bid" in this program and a bid on the Primarq
# marketplace are not quite the same entity.
# Here, bid.value is the *best* that an investor
# would be *willing* to bid; he will not actually bid
# so much unless forced up in a "bidding war" with
# another investor.
#
# The "winner" and "secondbid" attributes of an
# auction instance track the two best "bids" in the
# above sense; then, at the end of the day, winner.value
# is adjusted to reflect the best bid actually placed--
# each investor may have placed several actual bids or
# even none, but the best bid will be as calculated below.
#
# What we are technically modelling is therefore an iterated
# second-price sealed-bid auction, or iterated Vickrey auction
#   ( http://en.wikipedia.org/wiki/Vickrey_auction ).
# This should behave similarly to a standard "English auction"
# in the limit as timestep -> 0: you and I are at a cattle
# auction; we silently decide how high we want to bid;
# you bid 15 cents/lb, I bid 16 cents; with this new
# information, we might each adjust our private maximums.
#
# Even with a timestep as large as one day, I think we ought
# to see some cool results.


def howManyHEPS(house):
# The number of HEPs a homeowner wants to sell;
# i.e., 1/($10,000) * (desired money).

    # Guess at the claim percentage of a single HEP.
    p1 = 10000./valu(house)

    # The homeowner always retains at least 50% of the equity.
    if p1 > .5:
        return 0

    else:
        # Uniformly distributed, and at least 1:
        Min = 1
        Max = int(0.5/p1)
        return Min + int( random()*(Max-Min) )
                                

def isFinished(auct):
# Returns a Boolean:
# We call an auction 'finished' when
# nobofy has placed a new bid for a
# while ( a while = 1 day ).
#
# Alternatively, the auction closes if no one has met
# the seller's reserve price in 5 days of bidding.
# 
    B1 = (auct.oldWinner.bidder != None) and \
         (auct.isAsGood(auct.oldWinner, auct.winner) )

    B2 = (t - auct.startTime >= (5-1)*shortTimeStep) and \
         (auct.winner.bidder == None)
    
    return  B1 or B2


def newAuctions():
# Creates new auction instances as needed.
    for hep in heps:
        if hep.needsPrimary:
            hep.needsPrimary=False
            auctions.append(primaryClass(hep))
        if hep.needsSecondary:
            hep.needsSecondary=False
            auctions.append(secondaryClass(hep))

def runAuctions(auctions, valuations):
# Given a set of auctions and a matrix of valuations, find the
# winning bid for each auction.
    
    toRemove = set()

    # Each investor has limited available cash, so we will
    # prevent anyone from bidding more than
    #       cash - (winning bids already placed).
    # However, we want to minimize cases of "bidder remorse",
    # when an auction the bidder likes better comes up after an
    # auction on which he bid all his cash. We therefore sort
    # the auctions in descending order by general public
    # valuation, which should help.
    auctions.sort(
        key = lambda auct: valu(auct.hep.house),
        reverse = True)
        
    for auct in auctions:

        #initialize some variables
        auct.winner = auct.initializeWinner()
        secondbid = auct.winner.copy()
        auct.Tie = False

        #get bids
        for investor in investors:
            valu = investor.getValu(auct.hep.house) #House valuation.
            
            #print "day:",int(t*365),'',"valu:",valu#db
            value = auct.getBidValue(valu) # Best willing bid.

            bid = bidClass(investor, value)
            auct.bids.add(bid.copy())

            # Keep "winner", "secondbid" up to date:
            if auct.isAsGood(bid,auct.winner):
                secondbid = auct.winner.copy()
                auct.winner = bid.copy()
                if secondbid.value==auct.winner.value:
                    auct.tie = True
                else: auct.tie = False

            # The investor willing to bid second-best
            # might be queried *after* the best bidder
            # has been queried:
            elif auct.isAsGood(bid,secondbid):
                secondbid.value=bid.value
                
        # Is there a tie?
        if auct.tie:
            winners = [] # everybody who tied goes here.
            for bid in auct.bids:
                if bid.value == auct.winner.value:
                    winners.append(bid)

            # Pick one at random;
            # this is the lucky one who happened to place the actual
            # bid on the marketplace first; the others were
            # unwilling to outbid him.
            auct.winner = choice(winners)
            
        else:
            # If there was no tie, then the winning bidder might not
            # have gone fully as high (low) as he/she would have been
            # willing, but only a little bit higher (lower) than
            # the second-best bidder was willing to go.
            auct.winner.value = secondbid.value + auct.discrete

    return auctions

def finishAuctions(step):
# At the end of the day, check whether each auction is over or
# still ongoing. If ongoing, then update the oldWinner bid.
    for auct in auctions:

        # determine whether to close
        if isFinished(auct):

            #add to the list of auctions to close
            toRemove.add(auct)

        else:

            # End of the day, today's data becomes 'old'.
            auct.oldWinner = auct.winner.copy()
            auct.bids = set()

    # Close auctions
    for auct in toRemove:
        auct.close()
        
    # If there are any alive, we need the short time step.
    if len(auctions) > 0:
        step = shortTimeStep

    print "--------------------------------"#db

    return step

            
##################################################
#   Investor Valuations
##################################################

def valu(house):
# The valuation of a house from the point of view
# of the "general public". Just the expectation for now.
    hep = dataClass()
    hep.percentage=1
    hep.house=house
    return expectation(hep)
    

def payout(hep, salePrice = None, M0=None):
# The payout of a HEP upon its expiration,
#       P(S) = p(S - M0),
# where S is the final sale price,
# M0 is the initial mortgage amount,
# and p is the HEP claim percentage.
    if salePrice == None:
        return hep.percentage * \
               (hep.house.value - hep.house.M0)
    else:
        return hep.percentage * \
               (salePrice - M0)

def expectation(hep=None, pretty=False,**args):
# Expectation, in today's dollars, of the value of a HEP.
    '''Takes a HEP, or the following: mu, sigma, p, s0, M0.
    Optional: r, lambd.'''
    
    # Get parameters from keyword arguments,
    # or use global parameters.
    r = args.get('r', riskFree.Rate)
    lambd = args.get('lambd', sellDateLambda)

    # If 'hep' is a HEP object:
    if type(hep)==InstanceType:
            mu = hep.house.hood.dist.logMu
            sigma = hep.house.hood.dist.logSigma
            p = hep.percentage
            s0 = hep.house.value
            M0 = hep.house.M0

    # Use parameters from keyword arguments instead of
    # those from the hep, or if there is no hep.
    if 'mu' in args: mu = args['mu']
    if 'sigma' in args: sigma = args['sigma']
    if 'percentage' in args or 'p' in args:
        p = args.get('percentage',args.get('p'))
    if 's0' in args: s0 = args['s0']
    if 'M0' in args: M0 = args['M0']

    
            
    if mu+.5*sigma**2-lambd-r >= 0:
        riskFree.rate = r = mu+.5*sigma**2-lambd+0.0001
        
    expectation = p*lambd * \
           (-s0/(mu+.5*sigma**2-lambd-r)  \
            -M0/(lambd+r))

    # Print in readable format: e.g., $426,433.83
    if pretty:
        def pretty(money):
            if money < 1:
                s = str(round(money, 2))
                s += '0'*(4-len(s))
                return s
            
            E = floor(log(money, 1000))
            big = int(money/1000**E)
            
            if E ==0 :
                return str(big) + \
                       pretty( money-floor(money) )[1:]
            else    :
                s = pretty(money-big*1000**E)
                s = '0'*(3-len(s.split(',')[0])) + s
                return str(big) + ',' + s
            
        print "$" + pretty(expectation)

    return expectation


def HEP_distribution(house):
    
    t=0
    hood=house.hood.copy(t) #to do: make the copy function go deep
    hood.houses=set()
    house = houseClass(hood, distribution = False)
    
    value= 2250000  
    M0 = 0.8*value  

    def reset():
        t=0
        hood.appreciation={t:1}
        hood.lastTime=0
        house.value= value
        house.M0= M0
        house.sellDate=house.getSellDate()

    reset()
    hep = HEPclass(house);
    hep.percentage = 1

    v0 = expectation(hep)
    v1expect = mu = v0*exp(riskFree.rate)
    vList=[]
    
    real = 1
    n = numReals = 1000.
    while real<=numReals:
        if house.sellDate<=1:
            t=house.sellDate
            hood.appreciate(t)
            v1=(house.value-house.M0)*exp(riskFree.Rate*(1-t))

        else:
            t=1
            hood.appreciate(t)
            v1=expectation(hep)

        vList.append(v1)
        reset()
        real+=1

    mv = mean_and_variance(vList,True)
    mean = mv['mean']
    
    n=numReals; mu = v1expect
    sampleVar = mv['variance']
    Var = (sampleVar*(n-1) + n*mean**2  \
           -2*n*mean*mu + n*mu**2) /n

    def million(x):
        return str(round(x/10.**6,2))+" million."
    
    
    print "Initial Value: ", million(value)
    print "Mortgage:  ", million(M0)
    print
    print "Analytic Expectation: ", million(mu)
    print "Sample Mean: ", million(mean),mean
    print
    print "Variance from Expectation: ", million(Var)
    print "Sample Variance: ", million(sampleVar)
    print
    print "Standard Deviation from Expectation: ",\
          million(Var**.5)
    print "Sample Standard Deviation: ",\
          million(sampleVar**.5)

def getHoodDistribution(hood, t=0, resolution = 0.01):
# Simulates and returns the approximate cumulative
# distribution of the appreciation of a house in one year,
#        S1 / S0,
# given the parameters in the neighborhood.
#
# The distribution is in the form of a python function.
# It interpolates linearly when asked for a value at a
# finer resolution than recorded; e.g.,
# dist(0.90) returns the recorded 90th percentile value,
# but dist(0.903) returns 0.7*dist(0.90)+0.3*dist(0.91),
# assuming that resolution=0.01.

    hood = hood.copy(t)
    print hood.appreciation

    house = houseClass(hood,distribution = False)
    value0=hood.medianPrice
    house.M0 = value0*.71    # Not really important until
                             # we include default risk.

    # When using the chi^2 test to analyse a multinomial
    # distribution, the rule of thumb is to set the sample
    # size at least high enough that the expected number of
    # occurences in each bin is at least 5.
    #
    # We're not actually doing anything with the chi^2 test,
    # but we might in the future. Anyway, I took that rule and
    # and doubled it for good measure, and because computers
    # are fast.
    numData = int(1/resolution)
    multiplier = 10
    reals = numData*multiplier

    listX=range(reals)
    
    # Do a year's appreciation
    for realization in range(reals):
        t+=1
        house.value=value0

        hood.appreciate(t)

        # Record the ratio
        X=house.value/value0
        listX[realization] = X

    listX.sort()
    CD = range( numData)

    # The list is sorted, so the first 10 entries are in the bottom
    # percentile, the next 10 in the 2nd percentile, etc.
    for i in range(1,numData):
        CD[i] = listX[multiplier*i - 1]
    
    def dist(p):
    # Given a probability p, returns the approximate
    # 100p% value of the distribution.
    #
    # If p>.99 or p<.01 (for resolution=0.01), this function
    # simply returns dist(.99) or dist(.01) as appropriate.
    
        if p>1 or p<0:
            raise ValueError, "p is not a probability! You fool!"

        # If we have data with sufficient resolution, use it.
        p = (p/resolution)
        if p.is_integer():
            return CD[int(p)]
        elif p<1:
            return CD[1]
        elif p>len(CD)-1:
            return CD[-1]

        # Otherwise, take a weighted average of the endpoints
        # of the appropriate interval.
        else:
            c = int( ceil( p) )
            f = int( floor(p) )
            return CD[c]*(p-f) + \
                   CD[f]*(c-p)
    
    # Attach the mean, variance, log-mean, and log-variance
    # just in case they're needed.
    mv = mean_and_variance(listX)
    dist.mean = mv['mean']
    dist.variance = mv['variance']

    mv = mean_and_variance([log(i) for i in listX])
    dist.logMu = mv['mean']
    dist.logSigma = mv['variance']**.5

    return dist
    
        
t=0   
N = neighborhoodClass(**nbhdDict)
N.dist = getHoodDistribution(N)
H=houseClass(N)
H.value= N.medianPrice
H.M0 = N.medianPrice*1.1
E=expectation
    


#########################################
##   The main simulation procedure
##########################################

for realization in range(realizations):
    t = 0

    if sim==None: t= duration + 1
    
    else:
        # Make neighborhoods
        neighborhoods=set()
        for d in nbhdDicts:
            # Homes on the market right away?
            n = d.pop('initialHomes')

            nbhd = neighborhoodClass(**d)
            nbhd.dist = getHoodDistribution(nbhd)
            neighborhoods.add(nbhd)
            houses.add(houseClass(nbhd) for i in range(n))
            
        
        # make investors, auctions
        investors = set(investorClass() for i in range(numInvestors))
        auctions = list()
    
    while t<=duration:
        
        # If we're at the start of a new month,
        # reset time step size to 1 month.
        if (t/longTimeStep).is_integer():
            step = longTimeStep
            print t/longTimeStep

        
        # Check for houses sold
        if HomeSalesOn:
            toRemove = set()
            for house in houses:
                if t>= house.SellDate:
                    toRemove.add(house)
            for house in toRemove:
                house.remove()

        #Check for new houses
        for hood in neighborhoods:
            
            newHouses = hood.newHouses(t)
            houses.update(newHouses)
            
            #Create new HEPS
            for house in newHouses:
                n = howManyHEPS(house)
                newHEPS = set(HEPclass(house) for i in range(n))
                heps.update(newHEPs)

        # Update neighborhood appreciation and home values.
        # Notice that this applies to any newly-added homes
        # as well; i.e., the "initial value" of a home is
        # in fact the value as of neighborhood.lastTime.
        for nbhd in neighborhoods:
            nbhd.appreciate(t)

        # For now, the list of investors is fixed.
        # Later:
        # newInvestors = getNewInvestors()
        # do something.

        #Do investors know that the homeowner is planning to sell?
        for house in houses:
            if not house.knownSell:
                months = (house.sellDate - t)/12.
                p = 1-(months-1)**2 / 25.
                if random() < p:
                    house.knownSell = True

        # See if investors want to sell
        for investor in investors:
            forSale = investor.getSellList()

            for hep in forSale:
                hep.needsSecondary == True

        # Make new auctions
        newAuctions()

        # Get investor valuations and buying power.
        valuations = getValuations()

        # Run Auctions
        auctions = runAuctions(auctions, valuations)
        
        # Finish auctions for the day; if there are any still
        # ongoing, then we need the short timestep.
        step = finishAuctions(step)



        

        t += step



##########################################
#   Some testing and debugging stuff
##########################################

#meanGeo,
 #      sdGeo
 #               medianPrice,
   #              percent99,
    #             pHIP,
     #            pDisaster):
        
meanGeo=1.063








    
