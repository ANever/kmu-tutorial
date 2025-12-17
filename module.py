import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.io as pio
import requests
from lmfit import minimize, Parameters, Parameter, report_fit
pio.renderers.default = "notebook"
plt.style.use('ggplot')
# Jupyter Specifics
from IPython.display import HTML
from ipywidgets.widgets import interact, IntSlider, FloatSlider, Layout, ToggleButton

style = {'description_width': '100px'}
slider_layout = Layout(width='99%')


def ode_model(z, t, beta, sigma, gamma):
    """
    Reference https://www.idmod.org/docs/hiv/model-seir.html
    """
    S, E, I, R = z
    N = S + E + I + R
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I
    dRdt = gamma*I
    return [dSdt, dEdt, dIdt, dRdt]

def ode_solver(t, initial_conditions, params):
    initE, initI, initR, initN = initial_conditions
    beta, sigma, gamma = params['beta'].value, params['sigma'].value, params['gamma'].value
    initS = initN - (initE + initI + initR)
    res = odeint(ode_model, [initS, initE, initI, initR], t, args=(beta, sigma, gamma))
    return res
    
    
response = requests.get('https://api.rootnet.in/covid19-in/stats/history')
covid_history = response.json()['data']

keys = ['day', 'total', 'confirmedCasesIndian', 'confirmedCasesForeign', 'confirmedButLocationUnidentified',
        'discharged', 'deaths']
df_covid_history = pd.DataFrame([[d.get('day'), 
                                  d['summary'].get('total'), 
                                  d['summary'].get('confirmedCasesIndian'), 
                                  d['summary'].get('confirmedCasesForeign'),
                                  d['summary'].get('confirmedButLocationUnidentified'),
                                  d['summary'].get('discharged'), 
                                  d['summary'].get('deaths')] 
                                 for d in covid_history],
                    columns=keys)
df_covid_history = df_covid_history.sort_values(by='day')
df_covid_history['infected'] = df_covid_history['total'] - df_covid_history['discharged'] - df_covid_history['deaths']
df_covid_history['total_recovered_or_dead'] = df_covid_history['discharged'] + df_covid_history['deaths']


# ref: https://www.medrxiv.org/content/10.1101/2020.04.01.20049825v1.full.pdf
initN = 1380000000
# S0 = 966000000
initE = 1000
initI = 47
initR = 0
sigma = 1/5.2
gamma = 1/2.9
R0 = 4
beta = R0 * gamma
days = 112


def F():
    def ode_model(z, t, beta, sigma, gamma):
        """
        Reference https://www.idmod.org/docs/hiv/model-seir.html
        """
        S, E, I, R = z
        N = S + E + I + R
        dSdt = -beta*S*I/N
        dEdt = beta*S*I/N - sigma*E
        dIdt = sigma*E - gamma*I
        dRdt = gamma*I
        return [dSdt, dEdt, dIdt, dRdt]
    def ode_solver(t, initial_conditions, params):
        initE, initI, initR, initN = initial_conditions
        beta, sigma, gamma = params['beta'].value, params['sigma'].value, params['gamma'].value
        initS = initN - (initE + initI + initR)
        res = odeint(ode_model, [initS, initE, initI, initR], t, args=(beta, sigma, gamma))
        return res
    response = requests.get('https://api.rootnet.in/covid19-in/stats/history')
    covid_history = response.json()['data']

    keys = ['day', 'total', 'confirmedCasesIndian', 'confirmedCasesForeign', 'confirmedButLocationUnidentified',
            'discharged', 'deaths']
    df_covid_history = pd.DataFrame([[d.get('day'), 
                                      d['summary'].get('total'), 
                                      d['summary'].get('confirmedCasesIndian'), 
                                      d['summary'].get('confirmedCasesForeign'),
                                      d['summary'].get('confirmedButLocationUnidentified'),
                                      d['summary'].get('discharged'), 
                                      d['summary'].get('deaths')] 
                                     for d in covid_history],
                        columns=keys)
    df_covid_history = df_covid_history.sort_values(by='day')
    df_covid_history['infected'] = df_covid_history['total'] - df_covid_history['discharged'] - df_covid_history['deaths']
    df_covid_history['total_recovered_or_dead'] = df_covid_history['discharged'] + df_covid_history['deaths']

    # ref: https://www.medrxiv.org/content/10.1101/2020.04.01.20049825v1.full.pdf
    initN = 1380000000
    # S0 = 966000000
    initE = 1000
    initI = 47
    initR = 0
    sigma = 1/5.2
    gamma = 1/2.9
    R0 = 4
    beta = R0 * gamma
    days = 112
    
    params = Parameters()
    params.add('beta', value=beta, min=0, max=10)
    params.add('sigma', value=sigma, min=0, max=10)
    params.add('gamma', value=gamma, min=0, max=10)


    def main(initE, initI, initR, initN, beta, sigma, gamma, days, param_fitting):
        initial_conditions = [initE, initI, initR, initN]
        params['beta'].value, params['sigma'].value,params['gamma'].value = [beta, sigma, gamma]
        tspan = np.arange(0, days, 1)
        sol = ode_solver(tspan, initial_conditions, params)
        S, E, I, R = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
        
        # Create traces
        fig = go.Figure()
        if not param_fitting:
            fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines+markers', name='Susceptible'))
            fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines+markers', name='Exposed'))
        fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines+markers', name='Infected'))
        fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines+markers',name='Recovered'))
        if param_fitting:
            fig.add_trace(go.Scatter(x=tspan, y=df_covid_history.infected, mode='lines+markers',\
                                 name='Infections Observed', line = dict(dash='dash')))
            fig.add_trace(go.Scatter(x=tspan, y=df_covid_history.total_recovered_or_dead, mode='lines+markers',\
                                 name='Recovered/Deceased Observed', line = dict(dash='dash')))
        
        if days <= 30:
            step = 1
        elif days <= 90:
            step = 7
        else:
            step = 30
        
        # Edit the layout
        fig.update_layout(title='Simulation of SEIR Model',
                           xaxis_title='Day',
                           yaxis_title='Counts',
                           title_x=0.5,
                          width=900, height=600
                         )
        fig.update_xaxes(tickangle=-90, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
        if not os.path.exists("images"):
            os.mkdir("images")
        fig.show()
        
        observed_IR = df_covid_history.loc[:, ['infected', 'total_recovered_or_dead']].values
        
        tspan_fit_pred = np.arange(0, observed_IR.shape[0], 1)
        #params['beta'].value = result.params['beta'].value
        #params['sigma'].value = result.params['sigma'].value
        #params['gamma'].value = result.params['gamma'].value
        fitted_predicted = ode_solver(tspan_fit_pred, initial_conditions, params)
        fitted_predicted_IR = fitted_predicted[:, 2:4]
        print("Fitted MAE")
        print('Infected: ', np.mean(np.abs(fitted_predicted_IR[:days, 0] - observed_IR[:days, 0])))
        print('Recovered/Deceased: ', np.mean(np.abs(fitted_predicted_IR[:days, 1] - observed_IR[:days, 1])))

        print("\nFitted RMSE")
        print('Infected: ', np.sqrt(np.mean((fitted_predicted_IR[:days, 0] - observed_IR[:days, 0])**2)))
        print('Recovered/Deceased: ', np.sqrt(np.mean((fitted_predicted_IR[:days, 1] - observed_IR[:days, 1])**2)))
    

    interact(main, initE=IntSlider(min=0, max=100000, step=1, value=initE, description='initE', style=style, \
                               layout=slider_layout),
               initI=IntSlider(min=0, max=100000, step=1, value=initI, description='initI', style=style, \
                               layout=slider_layout),
               initR=IntSlider(min=0, max=100000, step=1, value=initR, description='initR', style=style, \
                               layout=slider_layout),
               initN=IntSlider(min=0, max=1380000000, step=10, value=initN, description='initN', style=style, \
                               layout=slider_layout),
               beta=FloatSlider(min=0, max=4, step=0.0001, value=beta, description='Infection rate', style=style, \
                                layout=slider_layout),
               sigma=FloatSlider(min=0, max=4, step=0.0001, value=sigma, description='Incubation rate', style=style, \
                                 layout=slider_layout),
               gamma=FloatSlider(min=0, max=4, step=0.0001, value=gamma, description='Recovery rate', style=style, \
                                 layout=slider_layout),
               days=IntSlider(min=0, max=600, step=7, value=days, description='Days', style=style, \
                              layout=slider_layout),
               param_fitting=ToggleButton(value=False, description='Fitting Mode', disabled=False, button_style='', \
             tooltip='Click to show fewer plots', icon='check-circle')
        );


    

    #return 



