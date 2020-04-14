"""
Countermeasures models.

TODO: Add some general description here.
n.b. some entries taken from PyMC3 documentation
"""
import logging
import math

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as T
from pymc3 import Model

from ..utils import geom_convolution, array_stats

log = logging.getLogger(__name__)


class BaseCMModel(Model):
    """
    Base class for building countermeasures models.

    Attributes
    ----------
    d : epimodel.pymc3_models.cm_effect.Loader
        A `Loader` containing data to be fed to the model
    plot_trace_vars : set[str]
        The PyMC3 variables to be plot by `plot_traces`
    trace : pymc3.backends.base.MultiTrace
        The PyMC3 `MultiTrace` object containing the sampling values from 
        the model.
    nRs : int
        The number of regions in the model's `Loader` (``len(d.Rs)``)
    nDs : int
        The number of dates in the model's `Loader` (``len(d.Ds)``)
    nCMs : int
        The number of countermeasures in the model's `Loader` (``len(d.CMs)``)

    Parameters
    ----------
    data : epimodel.pymc3_models.cm_effect.Loader
        A `Loader` containing data to be fed to the model
    model : pymc3.model.Model, optional
        Instance of `Model` that is supposed to be a parent for the new 
        instance. If None, context will be used. All variables defined 
        within instance will be passed to the parent instance. So that 
        ‘nested’ model contributes to the variables and likelihood factors 
        of parent model.
        Default: ""
    name : str, optional
        The name that will be used as a prefix for the names of all
        random variables defined within the model
        Default: ""

    """
    def __init__(self, data, model=None, name=""):
        super().__init__(name, model)
        self.d = data
        self.plot_trace_vars = set()
        self.trace = None

    def LogNorm(self, name, mean, log_var, plot_trace=True, hyperprior=None, **kwargs):
        """
        Create a lognorm variable, adding it to self as attribute.

        Variable will be created using
        ::
            pm.Lognormal(self.prefix + name, T.log(mean), log_var, **kwargs)

        Parameters
        ----------
        name : str
            Name of the variable
        mean : float
            Location parameter
        log_var : float
            Log of the variance (?)
        plot_trace : bool
            Whether or not to plot the trace of this variable in `plot_traces`
            (default: True)
        hyperprior
            TODO
        **kwargs
            Arguments to pass to `pm.Lognormal` when creating the variable

        Returns
        -------
        pymc3.distributions.continuous.Lognormal
        """
        if name in self.__dict__:
            log.warning(f"Variable {name} already present, overwriting def")
        if hyperprior:
            # TODO
            pass

        v = pm.Lognormal(self.prefix + name, T.log(mean), log_var, **kwargs)
        # self.__dict__[name] = v
        if plot_trace:
            self.plot_trace_vars.add(self.prefix + name)

        return v

    def Det(self, name, exp, plot_trace=True):
        """
        Create a deterministic variable, adding it to self as attribute.

        Variable will be created using
        ::
            pm.Deterministic(self.prefix + name, exp)

        Parameters
        ----------
        name : str
            Name of variable
        exp : Theano variable
            Value of variable
        plot_trace : bool
            Whether or not to plot the trace of this variable in `plot_traces`
            (default: True)

        Returns
        -------
        pymc3.model.Deterministic
        """
        if name in self.__dict__:
            log.warning(f"Variable {name} already present, overwriting def")
        v = pm.Deterministic(self.prefix + name, exp)
        # self.__dict__[name] = v
        if plot_trace:
            self.plot_trace_vars.add(self.prefix + name)
        return v

    @property
    def nRs(self):
        return len(self.d.Rs)

    @property
    def nDs(self):
        return len(self.d.Ds)

    @property
    def nCMs(self):
        return len(self.d.CMs)

    def plot_traces(self):
        """
        Plots traces of selected variables with ``pm.traceplot``.

        ::
            pm.traceplot(self.trace, var_names=list(self.plot_trace_vars))
        """
        assert self.trace is not None
        return pm.traceplot(self.trace, var_names=list(self.plot_trace_vars))

    def plot_CMReduction(self):
        """
        Plots a forest plot (90% credible interval) of the ``CMReduction`` 
        variable (i.e. the reduction to growth rate R0 of each of the
        countermeasures when active)
        
        ::
            pm.forestplot(
                self.trace, var_names=[self.prefix + "CMReduction"], credible_interval=0.9
            )
        """
        assert self.trace is not None
        return pm.forestplot(
            self.trace, var_names=[self.prefix + "CMReduction"], credible_interval=0.9
        )

    def print_CMReduction(self):
        """
        Prints statistics on the traces of ``CMReduction`` (i.e. the reduction to growth rate 
        R0 of each of the countermeasures when active) for each of the different
        countermeasures (using :meth:`epimodel.pymc3_models.utils.array_stats`)
        -- currently mean, standard deviation and (0.05th, 0.95th) quantiles.
        
        Example (TODO: this does not include std dev (?) )
        ::
             0 Masks over 60                            CMReduction_cumul 0.985 (0.947 .. 1)
             1 Asymptomatic contact isolation           CMReduction_cumul 0.926 (0.886 .. 0.961)
             2 Gatherings limited to 10                 CMReduction_cumul 0.911 (0.887 .. 0.934)
             3 Gatherings limited to 100                CMReduction_cumul 0.916 (0.893 .. 0.938)
             4 Gatherings limited to 1000               CMReduction_cumul 0.919 (0.896 .. 0.942)
             5 Business suspended - some                CMReduction_cumul 0.96 (0.934 .. 0.986)
             6 Business suspended - many                CMReduction_cumul 0.95 (0.926 .. 0.974)
             7 Schools and universities closed          CMReduction_cumul 0.933 (0.911 .. 0.958)
             8 General curfew - permissive              CMReduction_cumul 0.99 (0.966 .. 1)
             9 General curfew - strict                  CMReduction_cumul 0.952 (0.93 .. 0.975)
            10 Healthcare specialisation over 0.2       CMReduction_cumul 0.966 (0.93 .. 0.997)

        """
        varname = self.prefix + "CMReduction"
        for i, c in enumerate(self.d.CMs):
            print(f"{i:2} {c:30} {varname:20} {array_stats(self.trace[varname][:,i])}")

    def print_var_per_country(self, varname):
        """
        For each region, prints array statistics (using 
        :meth:`epimodel.pymc3_models.utils.array_stats`)
        for the trace of a given variable.

        Parameters
        ----------
        varname : str
            The name of the variable to print array statistics for
        """
        varname = self.prefix + varname
        for i, c in enumerate(self.d.Rs):
            print(
                f"{i:2} {self.d.rds[c].DisplayName:30} {varname:20} "
                f"{array_stats(self.trace[varname][i, ...])}"
            )

    def run(self, N, chains=2, cores=2):
        """
        Performs Bayesian inference on the model.

        Calls ``check_test_point()`` on the model, printing the results,
        then runs ``pm.sample`` on the model:

        ::
            pm.sample(N, chains=chains, cores=cores, init="adapt_diag")
        
        storing the resulting trace in `self.trace`.
        """
        print(self.check_test_point())
        with self:
            self.trace = pm.sample(N, chains=chains, cores=cores, init="adapt_diag")

class CMModelV2(BaseCMModel):
    """
    CM effect model V2 (lognormal prior)
    """

    def __init__(self, data, delay_mean=7.0):
        super().__init__(data)
        self.CMDelayProb, self.CMDelayCut = self.d.create_delay_dist(delay_mean)

    def build_reduction_var(self, scale=0.1):
        """
        Less informative prior for CM reduction, allows values >1.0
        """
        # [CM] How much countermeasures reduce growth rate
        return self.LogNorm("CMReduction", 1.0, scale, shape=(self.nCMs,))

    def build(self):
        """
        Build the model variables.
        """
        CMReduction = self.build_reduction_var()

        # Window of active countermeasures extended into the past
        Earlier_ActiveCMs = self.d.get_ActiveCMs(
            self.d.Ds[0] - pd.DateOffset(self.CMDelayCut), self.d.Ds[-1]
        )

        # [region, CM, day] Reduction factor for each CM,C,D
        ActiveCMReduction = (
            T.reshape(CMReduction, (1, self.nCMs, 1)) ** Earlier_ActiveCMs
        )

        # [region, day] Reduction factor from CMs for each C,D (noise added below)
        GrowthReduction = self.Det(
            "GrowthReduction", T.prod(ActiveCMReduction, axis=1), plot_trace=False
        )

        # [region, day] Convolution of GrowthReduction by DelayProb along days
        DelayedGrowthReduction = self.Det(
            "DelayedGrowthReduction",
            geom_convolution(GrowthReduction, self.CMDelayProb, axis=1)[
                :, self.CMDelayCut :
            ],
            plot_trace=False,
        )

        # [] Baseline growth rate (wide prior OK, mean estimates ~10% daily growth)
        BaseGrowthRate = self.LogNorm("BaseGrowthRate", 1.2, 2.3)

        # [region] Region growth rate
        # TODO: Estimate growth rate variance
        RegionGrowthRate = self.LogNorm(
            "RegionGrowthRate", BaseGrowthRate, 0.3, shape=(self.nRs,)
        )

        # [region] Region unreliability as common scale multiplier of its:
        # * measurements (measurement unreliability)
        # * expected growth noise
        # TODO: Estimate good prior (but can be weak?)
        RegionScaleMult = self.LogNorm("RegionScaleMult", 1.0, 1.0, shape=(self.nRs,))

        # [region, day] The ideal predicted daily growth
        PredictedGrowth = self.Det(
            "PredictedGrowth",
            T.reshape(RegionGrowthRate, (self.nRs, 1)) * DelayedGrowthReduction,
            plot_trace=False,
        )

        # [region, day] The actual (still hidden) growth rate each day
        # TODO: Estimate noise varince (should be small, measurement variance below)
        #       Miscalibration: too low: time effects pushed into CMs, too high: explains away CMs
        RealGrowth = self.LogNorm(
            "RealGrowth",
            PredictedGrowth,
            RegionScaleMult.reshape((self.nRs, 1)) * 0.1,
            shape=(self.nRs, self.nDs),
            plot_trace=False,
        )

        # [region, day] Multiplicative noise applied to predicted growth rate
        self.Det(
            "RealGrowthNoise", RealGrowth / PredictedGrowth, plot_trace=False,
        )

        # [region] Initial size of epidemic (the day before the start, only those detected; wide prior OK)
        InitialSize = self.LogNorm("InitialSize", 1.0, 10, shape=(self.nRs,))

        # [region, day] The number of cases that would be detected with noiseless testing
        # (Noise source includes both false-P/N rates and local variance in test volume and targetting)
        # (Since we ony care about growth rates and assume consistent testing, it is fine to ignore real size)
        Size = self.Det(
            "Size",
            T.reshape(InitialSize, (self.nRs, 1)) * RealGrowth.cumprod(axis=1),
            plot_trace=False,
        )

        # [region, day] Cummulative tested positives
        Observed = self.LogNorm(
            "Observed",
            Size,
            0.4,  # self.RegionScaleMult.reshape((self.nRs, 1)) * 0.4,
            shape=(self.nRs, self.nDs),
            observed=self.d.Confirmed,
            plot_trace=False,
        )

        # [region, day] Multiplicative noise applied to predicted growth rate
        # Note: computed backwards, since self.Observed needs to be a distribution
        self.Det(
            "ObservedNoise", Observed / Size, plot_trace=False,
        )


class CMModelV2g(CMModelV2):
    """
    CM effect model V2g (exp(-gamma) prior)
    """

    def build_reduction_var(self, alpha=0.5, beta=1.0):
        """
        CM reduction prior from ICL paper, only values <=1.0
        """
        # [CM] How much countermeasures reduce growth rate
        CMReductionGamma = pm.Gamma("CMReductionGamma", alpha, beta, shape=(self.nCMs,))
        return self.Det("CMReduction", T.exp((-1.0) * CMReductionGamma))


class CMModelV1(BaseCMModel):
    """
    CM effect model V1 (lognormal prior)
    """

    def __init__(self, data, delay_mean=7.0):
        super().__init__(data)
        self.CMDelayProb, self.CMDelayCut = self.d.create_delay_dist(delay_mean)
        self.observed_data = self.d.Confirmed
        self.noise_RealGrowth = 0.07
        self.noise_Observed = 0.3

    def build_reduction_var(self, scale=0.1):
        """
        Less informative prior for CM reduction, allows values >1.0
        """
        # [CM] How much countermeasures reduce growth rate
        return self.LogNorm("CMReduction", 1.0, scale, shape=(self.nCMs,))

    def build(self):
        """
        Build the model variables.
        """
        CMReduction = self.build_reduction_var()

        # [] Baseline growth rate (wide prior OK, mean estimates ~20% daily growth)
        BaseGrowthRate = self.LogNorm("BaseGrowthRate", 1.2, 2.0)

        # [region] Initial size of epidemic (the day before the start, only those detected; wide prior OK)
        InitialSize = self.LogNorm("InitialSize", 1.0, 10, shape=(self.nRs,))

        # [region] Region growth rate
        # TODO: Estimate growth rate variance
        RegionGrowthRate = self.LogNorm(
            "RegionGrowthRate", BaseGrowthRate, 0.3, shape=(self.nRs,)
        )

        # [region, CM, day] Reduction factor for each CM,C,D
        ActiveCMReduction = (
            T.reshape(CMReduction, (1, self.nCMs, 1)) ** self.d.ActiveCMs
        )

        # [region, day] Reduction factor from CMs for each C,D (noise added below)
        GrowthReduction = self.Det(
            "GrowthReduction", T.prod(ActiveCMReduction, axis=1), plot_trace=False
        )

        # [region, day] Convolution of GrowthReduction by DelayProb along days
        DelayedGrowthReduction = geom_convolution(
            GrowthReduction, self.CMDelayProb, axis=1
        )

        # Erase early DlayedGrowthRates in first ~10 days (would assume them non-present otherwise!)
        DelayedGrowthReduction = DelayedGrowthReduction[:, self.CMDelayCut :]

        # [region, day - CMDelayCut] The ideal predicted daily growth
        PredictedGrowth = self.Det(
            "PredictedGrowth",
            T.reshape(RegionGrowthRate, (self.nRs, 1)) * DelayedGrowthReduction,
            plot_trace=False,
        )

        # [region, day - CMDelayCut] The actual (still hidden) growth each day
        # TODO: Estimate noise varince (should be small, measurement variance below)
        #       Miscalibration: too low: time effects pushed into CMs, too high: explains away CMs
        RealGrowth = self.LogNorm(
            "RealGrowth",
            PredictedGrowth,
            self.noise_RealGrowth,
            shape=(self.nRs, self.nDs - self.CMDelayCut),
            plot_trace=False,
        )

        # [region, day] Multiplicative noise applied to predicted growth rate
        self.Det(
            "RealGrowthNoise", RealGrowth / PredictedGrowth, plot_trace=False,
        )

        # Below I assume plain exponentia growth of confirmed rather than e.g. depending on the remaining
        # susceptible opulation etc.

        # [region, day - CMDelayCut] The number of cases that would be detected with noiseless testing
        # (Noise source includes both false-P/N rates and local variance in test volume and targetting)
        # (Since we ony care about growth rates and assume consistent testing, it is fine to ignore real size)
        Size = self.Det(
            "Size",
            T.reshape(InitialSize, (self.nRs, 1)) * RealGrowth.cumprod(axis=1),
            plot_trace=False,
        )

        # [region, day - CMDelayCut] Cummulative tested positives
        Observed = pm.Lognormal(
            "Observed",
            Size,
            self.noise_Observed,
            shape=(self.nRs, self.nDs - self.CMDelayCut),
            observed=self.observed_data[:, self.CMDelayCut :],
        )

        # [region, day] Multiplicative noise applied to predicted growth rate
        # Note: computed backwards, since self.Observed needs to be a distribution
        self.Det(
            "ObservedNoise", Observed / Size, plot_trace=False,
        )


class CMModelV1g(CMModelV1):
    """
    CM effect model V1g (exp(-gamma) prior)
    """

    def build_reduction_var(self, alpha=0.5, beta=1.0):
        """
        CM reduction prior from ICL paper, only values <=1.0
        """
        # [CM] How much countermeasures reduce growth rate
        CMReductionGamma = pm.Gamma("CMReductionGamma", alpha, beta, shape=(self.nCMs,))
        return self.Det("CMReduction", T.exp((-1.0) * CMReductionGamma))
