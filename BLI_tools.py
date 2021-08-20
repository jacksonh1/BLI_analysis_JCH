import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
import plotly.express as px
plt.style.use('custom_standard')
# plt.style.use('custom_small')


def import_raw_data(datafile):
    '''
    Imports/parses the `RawDataX.xls` file that you export from the instrument computer. If you try to open this file in excel, it will throw an error because the extension is actually incorrect. the `RawDataX.xls` files are just simple `.tsv` files, not `.xls` files. (See for yourself by opening a real .xls file and the `RawDataX.xls` file in a text editor)

    Parameters
    ----------
    datafile : str
        The path to the raw `.xls` datafile (ex: /path/to/data/RawData0.xls
    
    Returns
    -------
    df : pandas DataFrame
        data in pandas DataFrame object. For each sample there should be a time column (`WellID_t`) and a BLI signal column (`WellID`). Ex: for sample well A1 there will be 2 columns in the dataframe: `A1_t`, `A1`
    '''
    # Store data in a DataFrame
    df = pd.read_csv(datafile, sep='\t')
    # remove the last column
    df = df.iloc[:,:-1]
    # make a list of the column names with the "unnamed" columns removed
    # Will leave just a list of the sample wells
    new_cols = [i for i in df.columns if 'Unnamed:' not in i]
    # make a list containing the new columns names by inserting names for the time columns
    new_cols2 = []
    for i in new_cols:
        new_cols2.append('t_{}'.format(i))
        new_cols2.append(i)
    # rename the columns
    df.columns = new_cols2
    return df


def plot_BLI_trace(sample_key, df, wells):
    fig, ax = plt.subplots(figsize = [15,7])
    for well in wells:
        s = sample_key[sample_key['tip well']==well][['tip','titrant']].values[0]
        title = '{}: {} + {}'.format(well, s[0], s[1])
        x = df['t_{}'.format(well)]
        y = df[well]
        ax.plot(x, y, label=title)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('response (nm)')
    ax.legend()
    plt.tight_layout()
    return ax


def enumerate_sample_wells(letters, numbers):
    """
    returns a list of well IDs (strings) that is a combination of the input letters/numbers
    Examples:
    >>> make_titration('A', range(1, 7))
    ['A1','A2','A3','A4','A5','A6']
    >>> make_titration('ABCD', [1,2,3])
    ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']
    """
    try: _ = iter(letters)
    except: letters = [letters]
    try: _ = iter(numbers)
    except: numbers = [numbers]
    assert len(letters) > 1 or len(numbers) > 1, "Must have at least one iterable of length > 1"
    wells = []
    for l in letters:
        for n in numbers:
            wells.append('{}{}'.format(l,n))
    return wells


def get_sample_wells(sample_key, tip, titrant):
    '''
    use `tip` and `titrant` to retrieve well IDs for given samples
    '''
    if tip == 'all':
        tip = list(sample_key['tip'].unique())
    else:
        tip = [tip] if isinstance(tip, str) else tip
    if titrant == 'all':
        titrant = list(sample_key['titrant'].unique())
    else:
        titrant = [titrant] if isinstance(titrant, str) else titrant
    # Use conditionals to filter the pandas dataframe
    d = sample_key[(sample_key['tip'].isin(tip)) & (sample_key['titrant'].isin(titrant))]
    # I could've done this all in one line but I think it's easier to understand like this
    # access "tip well" column and turn into a simple python list
    return d['tip well'].to_list()


class BLI_data_df:
    '''
    the class uses self.sample_wells to select subset of samples.
    By default, `sample_wells` is set to all samples in the input datafile.
    the methods starting with `set_wells...` will select a subset of the data to work with by changing the `sample_wells` variable.

    ---example---
    import the data
        data = BLI_tools.BLI_data2(datafile, sample_key)
    select samples by what's on the tip and titrant species
        data.set_wells_from_samplekey('pCare','Mena EVH1 only')
    check out the plots and choose times to normalize to
        data.plotly_plot_samples()
        data.zero_cols(150,160)
    plot normalized data
        data.pyplot_plot_samples(subtracted_data=True)
    '''

    def __init__(self, df, sample_key):
        wells = []
        for i in df.columns:
            if 't_' in i:
                continue
            wells.append(i)
        self.sample_wells = wells
        self.sample_key = sample_key
        self.df = df

    def set_wells_from_samplekey(self, tip='all', titrant='all'):
        # use the sample key to get the wells for this set of curves
        # if input arguments arent lists make them lists
        if tip == 'all':
            tip = list(self.sample_key['tip'].unique())
        else:
            tip = [tip] if isinstance(tip, str) else tip
        if titrant == 'all':
            titrant = list(self.sample_key['titrant'].unique())
        else:
            titrant = [titrant] if isinstance(titrant, str) else titrant
        self.tip = tip
        self.titrant = titrant
        d = self.sample_key[(self.sample_key['tip'].isin(tip)) & (self.sample_key['titrant'].isin(titrant))]
        self.sample_wells = d['tip well'].to_list()

    def set_wells_directly(self, cols):
        cols = [cols] if isinstance(cols, str) else cols
        self.sample_wells = cols

    def set_wells_by_enumerating(self, letters, numbers):
        """
        returns a list of well IDs (strings) that is a combination of the input letters/numbers
        Examples:
        >>> set_wells_by_enumerating('A', list(range(1, 7)))
        ['A1','A2','A3','A4','A5','A6']
        >>> set_wells_by_enumerating('ABCD', [1,2,3])
        ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']
        """
        try: _ = iter(letters)
        except: letters = [letters]
        try: _ = iter(numbers)
        except: numbers = [numbers]
        assert len(letters) > 1 or len(numbers) > 1, "Must have at least one iterable of length > 1"
        wells = []
        for l in letters:
            for n in numbers:
                wells.append('{}{}'.format(l,n))
        self.sample_wells = wells

    def find_nearest(self, array, value):
        idx = (np.abs(array-value)).argmin()
        return idx

    def baseline_zero(self, col, start, end):
        inds = self.find_nearest(self.df['t_{}'.format(col)], start)
        indf = self.find_nearest(self.df['t_{}'.format(col)], end)
        normed = self.df[col]-np.mean(self.df[col].iloc[inds:indf])
        return normed

    def zero_cols(self, start, end):
        new_cols = []
        for i in self.sample_wells:
            new_cols.append('t_{}'.format(i))
            new_cols.append(i)
        df2 = self.df.copy()
        df2 = df2[new_cols]
        for j in self.sample_wells:
            df2[j] = self.baseline_zero(j, start, end)
        self.baseline_subtracted_data = df2

    def plotly_plot_samples(self, subtracted_data=False):
        fig = go.Figure()
        if subtracted_data:
            dat = self.baseline_subtracted_data
        else:
            dat = self.df
        for well in self.sample_wells:
            s = self.sample_key[self.sample_key['tip well']==well][['tip','titrant']].values[0]
            title = '{}: {} + {}'.format(well, s[0], s[1])
            fig.add_trace(go.Scatter(x=dat['t_{}'.format(well)], y=dat[well], name=title))
        # fig.show()
        return fig

    def pyplot_plot_samples(self, subtracted_data=False):
        fig, ax = plt.subplots(figsize = [15,7])
        if subtracted_data:
            dat = self.baseline_subtracted_data
        else:
            dat = self.df
        for well in self.sample_wells:
            s = self.sample_key[self.sample_key['tip well']==well][['tip','titrant']].values[0]
            title = '{}: {} + {}'.format(well, s[0], s[1])
            x = dat['t_{}'.format(well)]
            y = dat[well]
            ax.plot(x, y, label=title)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('response (nm)')
        ax.legend()
        plt.tight_layout()
        return ax

    def set_assay_times(self, first_association_time, association_t, dissociation_t, dt, concentrations):
        '''
        sets the assay time parameters so that the binding signals can be calculated from integrating the BLI signal

        to calculate the binding curve, the function takes the average signal from the end of the association step and subtracts the average signal from the end of the subsequent dissociation step:
        binding signal = assn signal - dssn baseline signal

        Parameters
        ----------
        first_association_time : float or int
            the timepoint (in seconds) that you want to define as the first binding signal (usually the end of the first association step). Final signal will be the signal averaged over `first_association_time` and `first_association_time`+`dt`
        association_t : float or int
            the length (in seconds) of the association step
        dissociation_t : float or int
            the length (in seconds) of the dissociation step
        dt : float or int
            the length (in seconds) that the signal should be averaged over to determine the binding or baseline signal
        concentrations : list
            list of concentrations used in experiment
        '''
        self.first_association_time = first_association_time
        self.association_t = association_t
        self.dissociation_t = dissociation_t
        self.dt = dt
        self.concentrations = concentrations

    def binding_signal_preview(self, subtracted_data=False, legend=False):
        fig, ax = plt.subplots(figsize = [15,7])
        if subtracted_data:
            dat = self.baseline_subtracted_data
        else:
            dat = self.df
        for well in self.sample_wells:

            s = self.sample_key[self.sample_key['tip well']==well][['tip','titrant']].values[0]
            title = '{}: {} + {}'.format(well, s[0], s[1])
            x = dat['t_{}'.format(well)]
            y = dat[well]
            ax.plot(x, y, label=title)
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        t = self.first_association_time
        dt = self.dt
        for c in self.concentrations:

            ax.fill_between(x, 0, 1, where=(t<x) & (x<t+dt), color='r', alpha=0.5, transform=trans)
            base_t = t + self.dissociation_t
            ax.fill_between(x, 0, 1, where=(base_t<x) & (x<base_t+dt), color='k', alpha=0.3, transform=trans)
            t = base_t + self.association_t
        ax.set_xlabel('time (s)')
        ax.set_ylabel('response (nm)')
        if legend:
            ax.legend()
        plt.tight_layout()
        return ax

    def find_nearest(self, array, value):
        idx = (np.abs(array-value)).argmin()
        return idx

    def df_average(self, col, start):
    #     print(start, end)
        inds = self.find_nearest(self.df['t_{}'.format(col)], start)
        indf = self.find_nearest(self.df['t_{}'.format(col)], start+self.dt)
        return np.mean(self.df[col].iloc[inds:indf])

    def calculate_single_binding_curve(self, well):
        y = []
        t = self.first_association_time
        for conc in self.concentrations:
            sig = self.df_average(well, t)
            base_t = t + self.dissociation_t
            base = self.df_average(well, base_t)
            y.append(sig-base)
        #     print(conc,',',sig-base)
            t = base_t + self.association_t
        return y

    def generate_binding_curves(self):
        binding_curve_table = pd.DataFrame(columns=[str(i) for i in self.concentrations], index=self.sample_wells)
        for i in self.sample_wells:
            y = self.calculate_single_binding_curve(i)
            for j,c in enumerate(self.concentrations):
                binding_curve_table.loc[i,str(c)] = y[j]
        # return binding_curve_table.reset_index().rename(columns={'index':'well'})
        return binding_curve_table

    def generate_binding_curves2(self):
        binding_curve_table = pd.DataFrame(index=[str(i) for i in self.concentrations], columns=self.sample_wells)
        for i in self.sample_wells:
            y = self.calculate_single_binding_curve(i)
            for j,c in enumerate(self.concentrations):
                binding_curve_table.loc[str(c),i] = y[j]
        return binding_curve_table