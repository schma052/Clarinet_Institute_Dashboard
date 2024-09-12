import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pandasql as psql
from pandasql import sqldf
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import statsmodels.api as sm





# Set global font size for matplotlib
plt.rcParams.update({'font.size': 13})  # You can adjust the size here as needed

# Givens
    # List of all possible instruments
instruments = ['clarinet', 'oboe', 'flute', 'recorder', 'saxophone', 'brass',
                'trumpet', 'bassoon', 'frenchhorn', 'woodwind', 'tuba', 'euphonium',
                'soundfiles', 'string']

# Streamlit page configuration
st.title('_Online Music Retailer Dashboard_')
st.markdown('_**by Matt Schorr**_')
st.markdown('**Please download the two .csv datasets from my GitHub: :blue[https://github.com/schma052/Online_Music_Retailer_Dashboard]**')

# File uploaders
uploaded_file_sales = st.file_uploader("Upload CSV of Physical Sales (Weebly)", type='csv')
uploaded_file_customer = st.file_uploader("Upload CSV of Digital Sales (Payhip)", type='csv')

if uploaded_file_sales is not None and uploaded_file_customer is not None:
    # Processing the physical sales data
    columns = ['Date', 'Sales', 'Orders']  # Assuming the date is the first column
    df = pd.read_csv(uploaded_file_sales, names=columns, skiprows=16, sep=',', index_col=False)
    df = df[:-16]  # Trim the last 16 rows
    df = df[['Date', 'Sales']]
    # Rename the 'Sales' column to 'Physical Sales'
    df.rename(columns={'Sales': 'Physical Sales'}, inplace=True)
    # Convert the 'Date' column to datetime type
    df['Date'] = pd.to_datetime(df['Date'])
    # Sort the DataFrame by the 'Date' column
    df = df.sort_values('Date')

    # Processing the digital sales data
    payhip = pd.read_csv(uploaded_file_customer, sep=',')
    payhip = payhip[['Date', 'Amount Net']]
    # Rename the 'Amount Net' column to 'Digital Sales'
    payhip.rename(columns={'Amount Net': 'Digital Sales'}, inplace=True)
    # Convert the 'Date' column to datetime type
    payhip['Date'] = pd.to_datetime(payhip['Date'])
    # Sort the DataFrame by the 'Date' column
    payhip = payhip.sort_values('Date')
    # Group by 'Date' and sum the 'Sales'
    payhip_grouped = payhip.groupby('Date').sum()
    # Reset the index to turn 'Date' back into a column
    payhip_grouped.reset_index(inplace=True)

    # Merging the DataFrames on 'Date'
    merged_df = pd.merge(df, payhip_grouped, on='Date', how='outer')
    # Convert 'Sales' columns to numeric, coercing errors to NaN (or 0 if you prefer)
    merged_df['Physical Sales'] = pd.to_numeric(merged_df['Physical Sales'], errors='coerce').fillna(0)
    merged_df['Digital Sales'] = pd.to_numeric(merged_df['Digital Sales'], errors='coerce').fillna(0)

    # Create a new column by adding the two sales columns
    merged_df['Total Sales'] = (merged_df['Physical Sales'] + merged_df['Digital Sales'])
    # Rename the 'Total Sales' column to 'Sales'
    merged_df.rename(columns={'Total Sales': 'Sales'}, inplace=True)
    
    # Convert 'Date' to string format for display purposes only
    merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
     
    # Define a function to use pandasql
    pysqldf = lambda q: sqldf(q, globals())
       

    # SQL query adjusted for pandasql
    sqlquery = """
    SELECT
        Date,
        strftime('%Y', Date) AS sales_year,
        strftime('%m', Date) AS sales_month,
        SUM(`Digital Sales`) AS `Digital Sales`,
        SUM(`Physical Sales`) AS `Physical Sales`,
        SUM(`Digital Sales` + `Physical Sales`) AS `Total Sales`
    FROM
        merged_df
    GROUP BY
        strftime('%Y', Date),
        strftime('%m', Date)
    ORDER BY
        sales_year,
        sales_month;
    """
    
    # Execute the query
    Merged_df = pysqldf(sqlquery)
    
    
    qry= """
        Select SUM(`Physical Sales`) AS `Total Gross Physical Revenue`, SUM(`Digital Sales`) AS `Total Net Digital Revenue`, SUM(`Sales`) AS `Total Revenue`, COUNT(DISTINCT strftime('%Y', Date)) AS `Num. of Years`
        FROM merged_df
    """
    qry_df= pysqldf(qry)

    # Display the merged DataFrame
    st.markdown("**Daily Sales:**")
    st.area_chart(merged_df.set_index('Date')[['Physical Sales', 'Digital Sales']], stack = True)
    st.markdown(":blue[Notice the slow fade into the darker digital sales]")
    st.dataframe(qry_df, hide_index = True)
    st.dataframe(merged_df.describe())
    st.markdown(":blue[Digital sales improves revenue stability.]")
    
    #Merged_df['Date'] = pd.to_datetime(Merged_df['Date'])  # Ensure the date is in datetime format

    # Line plot of sales over time
    st.markdown("**Monthly Sales:**")
    st.line_chart(Merged_df.set_index('Date')[['Physical Sales', 'Digital Sales', 'Total Sales']])
    st.markdown("Descritive Stats:")
    st.dataframe(Merged_df[['Date', 'Physical Sales', 'Digital Sales', 'Total Sales']].describe())
    
# Seasonality    
if uploaded_file_sales is not None and uploaded_file_customer is not None:
    uploaded_file_customer.seek(0)  # Reset the file pointer to the start of the file every time before reading       
    uploaded_file_sales.seek(0)
    
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    # SQL query for monthly data
    query1 = """
    SELECT 
        strftime('%Y', Date) as Year, 
        strftime('%m', Date) as Month,
        SUM(`Digital Sales`) AS Digital_Sales,
        SUM(`Physical Sales`) AS Physical_Sales,
        SUM(Sales) AS Total_Sales,
        AVG(`Digital Sales`) AS Avg_Digital_Sales,
        AVG(`Physical Sales`) AS Avg_Physical_Sales,
        AVG(Sales) AS Avg_Sales
    FROM merged_df
    GROUP BY Month
    """

    # SQL query for day of the week data
    query2 = """
    SELECT 
        CASE strftime('%w', Date) 
            WHEN '0' THEN 'Sunday'
            WHEN '1' THEN 'Monday'
            WHEN '2' THEN 'Tuesday'
            WHEN '3' THEN 'Wednesday'
            WHEN '4' THEN 'Thursday'
            WHEN '5' THEN 'Friday'
            WHEN '6' THEN 'Saturday'
        END AS DayOfWeek,
        SUM(`Digital Sales`) AS Digital_Sales,
        SUM(`Physical Sales`) AS Physical_Sales,
        SUM(Sales) AS Total_Sales,
        AVG(`Digital Sales`) AS Avg_Digital_Sales,
        AVG(`Physical Sales`) AS Avg_Physical_Sales,
        AVG(Sales) AS Avg_Sales
    FROM merged_df
    GROUP BY DayOfWeek
    """
    monthly_data = psql.sqldf(query1, locals())
    weekly_data = psql.sqldf(query2, locals())
    
    # Define a mapping from day names to weekday numbers
    day_order = {
        'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
        'Thursday': 4, 'Friday': 5, 'Saturday': 6
    }

    # Map the 'DayOfWeek' to a sortable numerical value using the day_order
    weekly_data['DayNum'] = weekly_data['DayOfWeek'].map(day_order)

    # Sort the DataFrame based on the 'DayNum'
    weekly_data_sorted = weekly_data.sort_values('DayNum')

    kpi_daily = """
    
    SELECT
        strftime('%Y', Date) as Year,
        strftime('%m', Date) as Month,
        AVG(`Digital Sales`) AS `Digital Sales Avg`, 
        AVG(`Physical Sales`) AS `Physical Sales Avg`,
        AVG(`Sales`) AS `Sales Avg`
    FROM merged_df
    GROUP BY Month, Year
    """
    
    kpi_monthly = """
    
    SELECT
        strftime('%Y', Date) as Year,
        strftime('%w', Date) as Day,
        AVG(`Digital Sales`) AS `Digital Sales Avg`, 
        AVG(`Physical Sales`) AS `Physical Sales Avg`,
        AVG(`Sales`) AS `Sales Avg`
    FROM merged_df
    GROUP BY Day, Year
    """
    kpi_monthly = psql.sqldf(kpi_monthly, locals())
    kpi_daily = psql.sqldf(kpi_daily, locals())
    
    #Plotting
    
    # Convert DataFrames to HTML and exclude the index
    html_kpi_daily = kpi_daily.to_html(index=False)
    html_kpi_monthly = kpi_monthly.to_html(index=False)

    #st.markdown("**AVG Sales by Month:**", unsafe_allow_html=True)
    #st.markdown(html_kpi_daily, unsafe_allow_html=True)  # Display DataFrame without index

    #st.markdown("**AVG Sales by Day:**", unsafe_allow_html=True)
    #st.markdown(html_kpi_monthly, unsafe_allow_html=True)  # Display DataFrame without index
    
    # Plotting the Monthly Sales
    # no longer need the 'DayNum' column, drop it
    #weekly_data_sorted = weekly_data_sorted.drop(columns=['DayNum'])

    
    def plot_monthly_sales(monthly_data):
        fig = go.Figure()

        # Adding bar chart for physical sales
        fig.add_trace(go.Bar(
            x=monthly_data['Month'],
            y=monthly_data['Physical_Sales'],
            name='Physical Sales',
            marker_color='lightblue'
           
        ))

        # Adding bar chart for digital sales
        fig.add_trace(go.Bar(
            x=monthly_data['Month'],
            y=monthly_data['Digital_Sales'],
            name='Digital Sales',
            marker_color='blue'
            
        ))

        # Adding line chart for average physical sales
        fig.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['Avg_Physical_Sales'],
            name='Avg Physical Sales',
            mode='lines',
            line=dict(color='cyan', dash='dashdot'),
            yaxis='y2'
        ))

        # Adding line chart for average digital sales
        fig.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['Avg_Digital_Sales'],
            name='Avg Digital Sales',
            mode='lines',
            line=dict(color='navy', dash='dashdot'),
            yaxis='y2'
        ))

        # Update layout to create a dual y-axis graph
        fig.update_layout(
            title='Monthly Seasonality:',
            xaxis_title='Month',
            yaxis=dict(
                title='Total Sales',
                side='left'  # primary y-axis for sales
            ),
            yaxis2=dict(
                title='Average Sales',
                overlaying='y',
                side='right',  # secondary y-axis for averages
                showgrid=False  # Optionally hide the gridlines for the secondary y-axis
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )

        return fig

    # Display the plot in Streamlit
    monthly_fig = plot_monthly_sales(monthly_data)
    st.plotly_chart(monthly_fig, use_container_width=True)
    st.markdown(":blue[A semi-annual season can be clearly observed for physical products, while digital products fluctuate with the quarter.]")
    
    
    def plot_weekly_sales(weekly_data_sorted):
        fig = go.Figure()

        # Adding bar chart for physical sales
        fig.add_trace(go.Bar(
            x=weekly_data_sorted['DayNum'],
            y=weekly_data_sorted['Physical_Sales'],
            name='Physical Sales',
            marker_color='lightblue'
           
        ))

        # Adding bar chart for digital sales
        fig.add_trace(go.Bar(
            x=weekly_data_sorted['DayNum'],
            y=weekly_data_sorted['Digital_Sales'],
            name='Digital Sales',
            marker_color='blue'
            
        ))

        # Adding line chart for average physical sales
        fig.add_trace(go.Scatter(
            x=weekly_data_sorted['DayNum'],
            y=weekly_data_sorted['Avg_Physical_Sales'],
            name='Avg Physical Sales',
            mode='lines',
            line=dict(color='cyan', dash='dashdot'),
            yaxis='y2'
        ))

        # Adding line chart for average digital sales
        fig.add_trace(go.Scatter(
            x=weekly_data_sorted['DayNum'],
            y=weekly_data_sorted['Avg_Digital_Sales'],
            name='Avg Digital Sales',
            mode='lines',
            line=dict(color='navy', dash='dashdot'),
            yaxis='y2'
        ))

        # Update layout to create a dual y-axis graph
        fig.update_layout(
            title='Daily Seasonality:',
            xaxis_title='Days',
            yaxis=dict(
                title='Total Sales',
                side='left'  # primary y-axis for sales
            ),
            yaxis2=dict(
                title='Average Sales',
                overlaying='y',
                side='right',  # secondary y-axis for averages
                showgrid=False  # Optionally hide the gridlines for the secondary y-axis
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )

        return fig

    # Display the plot in Streamlit
    weekly_fig = plot_weekly_sales(weekly_data_sorted)
    st.plotly_chart(weekly_fig, use_container_width=True)
    st.markdown(":blue[We see the positive impacts from Tuesday morning advertising on both physical and digital products. When digging deeper, serperate from this dashboard, using statistical methods including the distributed lag model, its infinite counterpart, multinomial logit, mixed effects models, and VAR, all agree that Tuesday's advertising is significantly impacting sales very similarly to what can be seen above.]")
    st.text(" ")
    
# Adv Impact on Items (Adv Occurs Tuesday Morning)
if uploaded_file_sales is not None and uploaded_file_customer is not None:
    uploaded_file_customer.seek(0)  # Reset the file pointer to the start of the file every time before reading
    # Items are in _customer
    PH_df = pd.read_csv(uploaded_file_customer, sep = ',')  
    
    # Rename the 'Amount Net' column to 'Sales'
# Rename the 'Amount Net' column to 'Sales'
    PH_df.rename(columns={'Items In Cart': 'Items', 'Country Name': 'Country', 
                          'Unsubscribed From Email Updates': 'Email Unsub'}, inplace=True)
    
    pysqldf = lambda q: sqldf(q, globals())  

    kw_day_q = """
SELECT
    Day,
    SUM(clarinet) AS clarinet,
    SUM(oboe) AS oboe,
    SUM(flute) AS flute,
    SUM(recorder) AS recorder,
    SUM(saxophone) AS saxophone,
    SUM(brass) AS brass,
    SUM(trombone) AS trombone,
    SUM(bassoon) AS bassoon,
    SUM(trumpet) AS trumpet,
    SUM(frenchhorn) AS frenchhorn,
    SUM(woodwind) AS woodwind,
    SUM(tuba) AS tuba,
    SUM(euphonium) AS euphonium,
    SUM(cello) AS cello,
    SUM(soundfiles) AS soundfiles,
    SUM(string) AS string
FROM
    (SELECT
        Date,
        strftime('%w', Date) AS Day,
        CASE WHEN LOWER(Items) LIKE '%clarinet%' THEN 1 ELSE 0 END AS clarinet,
        CASE WHEN LOWER(Items) LIKE '%oboe%' THEN 1 ELSE 0 END AS oboe,
        CASE WHEN LOWER(Items) LIKE '%flute%' THEN 1 ELSE 0 END AS flute,
        CASE WHEN LOWER(Items) LIKE '%recorder%' THEN 1 ELSE 0 END AS recorder,
        CASE WHEN LOWER(Items) LIKE '%saxophone%' THEN 1 ELSE 0 END AS saxophone,
        CASE WHEN LOWER(Items) LIKE '%brass%' THEN 1 ELSE 0 END AS brass,
        CASE WHEN LOWER(Items) LIKE '%trombone%' THEN 1 ELSE 0 END AS trombone,
        CASE WHEN LOWER(Items) LIKE '%bassoon%' THEN 1 ELSE 0 END AS bassoon,
        CASE WHEN LOWER(Items) LIKE '%trumpet%' THEN 1 ELSE 0 END AS trumpet,
        CASE WHEN LOWER(Items) LIKE '%french_horn%' THEN 1 ELSE 0 END AS 'frenchhorn',
        CASE WHEN LOWER(Items) LIKE '%ww%' THEN 1 ELSE 0 END AS woodwind,
        CASE WHEN LOWER(Items) LIKE '%tuba%' THEN 1 ELSE 0 END AS tuba,
        CASE WHEN LOWER(Items) LIKE '%euphonium%' THEN 1 ELSE 0 END AS euphonium,
        CASE WHEN LOWER(Items) LIKE '%cello%' THEN 1 ELSE 0 END AS cello,
        CASE WHEN LOWER(Items) LIKE '%sound_files%' THEN 1 ELSE 0 END AS 'soundfiles',
        CASE WHEN LOWER(Items) LIKE '%string%' THEN 1 ELSE 0 END AS string
    FROM PH_df
    WHERE 
        LOWER(Email) NOT IN ('brahms23@yahoo.com', 'brahms23@yahoo.com', 'brahms23@gmail.com')
        OR LOWER(Items) NOT IN ('aa clarinet 1 evaluation', 'aa clarinet 2 evaluation')
        OR LOWER(`Payment Type`) NOT IN ('free')
    )
GROUP BY Day
"""
    dkw_df = pysqldf(kw_day_q)
  

    dr_country_q = """
SELECT
    Day,
    Country,
    SUM(`Amount Net`) AS `Digital Net Revenue`
FROM
    (SELECT
        Date,
        strftime('%w', Date) AS Day,
        `Amount Net`,
        Country
    FROM PH_df
    WHERE 
        LOWER(Email) NOT IN ('brahms23@yahoo.com', 'brahms23@yahoo.com', 'brahms23@gmail.com')
        OR LOWER(Items) NOT IN ('aa clarinet 1 evaluation', 'aa clarinet 2 evaluation')
        OR LOWER(`Payment Type`) NOT IN ('free')
    )
GROUP BY Country, Day
"""
    cdr_df = pysqldf(dr_country_q)
    
   # Display Advertising Cycle in Different Countries
    st.markdown("**Advertising Cycle in Different Countries:**")
   
    # Define the function to filter countries with sales on every day of the week
    def get_countries_with_sales_every_day(dataframe):
        # Group by both Country and Day, then calculate total digital net revenue
        country_sales_by_day = dataframe.groupby(['Country', 'Day'])['Digital Net Revenue'].sum().reset_index()

        # Check if each country has sales for all 7 days (from 0 to 6)
        countries_with_full_week_sales = country_sales_by_day.groupby('Country')['Day'].nunique().reset_index()
        countries_with_full_week_sales = countries_with_full_week_sales[countries_with_full_week_sales['Day'] >= 5]

        # Filter the original dataframe to include only these countries
        filtered_dataframe = dataframe[dataframe['Country'].isin(countries_with_full_week_sales['Country'])]
        return filtered_dataframe

    # Define the function for calculating deciles
    def get_deciles(dataframe):
        # Group by both Country and Day, then calculate total digital net revenue
        country_revenue_by_day = dataframe.groupby(['Country', 'Day'])['Digital Net Revenue'].sum().reset_index()

        # Calculate decile rankings based on the summed revenue by country
        total_revenue_by_country = country_revenue_by_day.groupby('Country')['Digital Net Revenue'].sum().reset_index()
        total_revenue_by_country['Quintile'] = pd.qcut(total_revenue_by_country['Digital Net Revenue'], 5, labels=[
            'Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        # Merge the deciles back into the country_revenue_by_day dataframe
        detailed_revenue = country_revenue_by_day.merge(total_revenue_by_country[['Country', 'Quintile']], on='Country',
                                                        how='left')
        return detailed_revenue

    # Apply the filtering function to limit countries that have sold items every day of the week
    filtered_cdr_df = get_countries_with_sales_every_day(cdr_df)

    # Apply the deciles function to the filtered data
    country_deciles = get_deciles(filtered_cdr_df)

    # Dropdown for selecting deciles
    selected_decile = st.selectbox('Select Quintile', ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # Filter data based on selection
    filtered_data = country_deciles[country_deciles['Quintile'] == selected_decile]

    # Show the bar chart if there is data to display
    if not filtered_data.empty:
        st.bar_chart(filtered_data[['Day', 'Digital Net Revenue', 'Country']], x='Day', y='Digital Net Revenue', color='Country', stack=False, use_container_width=True)
    else:
        st.markdown("No data to display. Please adjust your selection.")

    # Display Items Through Advertising Cycle
    st.write("**Items Through Advertising Cycle:**")

    # First, set the 'Day' column as the index if it's not already
    if 'Day' not in dkw_df.columns:
        dkw_df = dkw_df.set_index('Day')

    # Number of columns to display checkboxes in a single row
    num_columns = 5  # You can adjust this number based on your preference or screen size
    columns = st.columns(num_columns)

    # Create a dictionary to hold the checkbox state for each instrument, placed in a horizontal layout
    selected_instruments = {}
    for i, instrument in enumerate(instruments):
        with columns[i % num_columns]:  # This will distribute checkboxes across the columns
            selected_instruments[instrument] = st.checkbox(instrument, True)

    # Filter the DataFrame based on selected instruments
    selected_columns = [inst for inst, selected in selected_instruments.items() if selected]

    # Show the bar chart if there are any selected columns
    if selected_columns:
        st.bar_chart(dkw_df[selected_columns], stack=False)
    else:
        st.markdown(":red[Please select at least one instrument to display the chart.]")

    st.markdown(":blue[Advertising impact changes depending on the item, item type, and the country.]")
    st.markdown(" ")
    
# Table for Items 
if uploaded_file_sales is not None and uploaded_file_customer is not None:
    uploaded_file_customer.seek(0)  # Reset the file pointer to the start of the file every time before reading

    payhip_df = pd.read_csv(uploaded_file_customer, sep = ',')
    
    # Function to use pandasql
    pysqldf = lambda q: sqldf(q, globals())    
    query = """
    SELECT *
    FROM payhip_df
    WHERE Email NOT IN ('brahms23@yahoo.com', 'Brahms23@yahoo.com', 'brahms23@gmail.com', 'Brahms23@gmail.com')
    OR `Items In Cart` NOT IN ('AA Clarinet 1 evaluation', 'AA Clarinet 2 evaluation')
    """
    result = pysqldf(query)
    
    # Naming
    # Rename the 'Amount Net' column to 'Sales'
    result.rename(columns={'Items In Cart': 'Items'}, inplace=True)
    result.rename(columns={'Country Name': 'Country'}, inplace=True)
    result.rename(columns={'Unsubscribed From Email Updates': 'Email Unsub'}, inplace=True)
    
    
    # Remove the words 'archives' and 'sheet music' from 'Items'
    result['Items'] = result['Items'].str.replace(',', '  ', regex=False)
    # Regular expression to remove variations of 'digital download'
    result['Items'] = result['Items'].str.replace(r'\b[dD]igital\s[dD]ownload\b', '', regex=True)
    result['Items'] = result['Items'].str.replace('-', '', regex=False)
    result['Items'] = result['Items'].str.replace('Sheet Music', '', regex=False)
    result['Items'] = result['Items'].str.replace('Volume', 'vol.', regex=False)
    result['Items'] = result['Items'].str.replace('Volumes', 'vols.', regex=False)
    result['Items'] = result['Items'].str.replace('Vol.', 'vol.', regex=False)
    result['Items'] = result['Items'].str.replace('Vols.', 'vols.', regex=False)    
    result['Items'] = result['Items'].str.replace('All 4', 'all 4', regex=False)
    # Explode the 'Items' column into separate rows if needed
    result['Items'] = result['Items'].str.split(r'\s{3}(?=[A-Z])')
    result = result.explode('Items').reset_index(drop=True)
    
    # Normalize item names by trimming spaces and converting to lower case
    result['Items'] = result['Items'].str.strip().str.lower()
    result['Items'] = result['Items'].str.replace('  ', ' ', regex=False)
    

    #result['Items'] = result['Items'].str.replace('  ', ' ', regex=False)
    result['Items'] = result['Items'].str.replace('clarinet sound files vol. 1', 'sound files for clarinet vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('bassoon sound files vol. 1', 'sound files for bassoon vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('clarinet sound files vol. 2', 'sound files for clarinet vol. 2', regex=False)
    result['Items'] = result['Items'].str.replace('saxophone sound files vol. 1', 'sound files for saxophone vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('trumpet sound files vol. 1', 'sound files for trumpet vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('french horn sound files vol. 1', 'sound files for french horn vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('oboe sound files vol. 1', 'sound files for oboe vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('sound files for ww5 - clarinet', 'clarinet - sound files for ww5', regex=False)
    result['Items'] = result['Items'].str.replace('flute sound files vol. 1', 'sound files for flute vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('recorder sound files vol. 1', 'sound files for recorder vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('brass quintet-quartet sound files vol. 1', 'sound files for brass quintet-quartet vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('bassoon - sound files for ww5', 'sound files for ww5 - bassoon', regex=False)
    result['Items'] = result['Items'].str.replace('sound files for ww5 - flute', 'flute - sound files for ww5', regex=False)
    result['Items'] = result['Items'].str.replace('sound files for ww5 - french horn', 'french horn - sound files for ww5', regex=False)
    result['Items'] = result['Items'].str.replace('sound files for ww5 - oboe', 'oboe - sound files for ww5', regex=False)
    result['Items'] = result['Items'].str.replace('vol.s', 'vols.', regex=False)
    result['Items'] = result['Items'].str.replace('bassoon vol. 1', 'bassoon archive vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('bassoon vol. 3', 'bassoon archive vol. 3', regex=False)
    result['Items'] = result['Items'].str.replace('cello archive vols. 1 - 3', 'cello archive vols. 1-3', regex=False)
    result['Items'] = result['Items'].str.replace('trombone archive vol. 1', 'trombone archive', regex=False)
    result['Items'] = result['Items'].str.replace('trombone archive', 'trombone', regex=False)
    result['Items'] = result['Items'].str.replace('trombone-euphonium archive vol. 1', 'trombone', regex=False)
    result['Items'] = result['Items'].str.replace('cello archive vol. 1', 'cello vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('clarinet vol. 1', 'clarinet archive vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('clarinet vol. 2', 'clarinet archive vol. 2', regex=False)
    result['Items'] = result['Items'].str.replace('clarinet vol. 3', 'clarinet archive vol. 3', regex=False)
    result['Items'] = result['Items'].str.replace('clarinet vol. 4', 'clarinet archive vol. 4', regex=False)
    result['Items'] = result['Items'].str.replace('clarinet vols. 1 and 2', 'clarinet archive vols. 1 and 2', regex=False)
    result['Items'] = result['Items'].str.replace('clarinet vols. 3 and 4', 'clarinet archive vols. 3 and 4', regex=False)
    result['Items'] = result['Items'].str.replace('complete oboe archive vols. 1 2 and 3', 'oboe archive vols. 1 2 and 3', regex=False)
    result['Items'] = result['Items'].str.replace('oboe vols. 1 2 and 3', 'oboe archive vols. 1 2 and 3', regex=False)
    result['Items'] = result['Items'].str.replace('flute vol. 1', 'flute archive vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('flute vol. 2', 'flute archive vol. 2', regex=False)
    result['Items'] = result['Items'].str.replace('flute vol. 3', 'flute archive vol. 3', regex=False)
    result['Items'] = result['Items'].str.replace('flute vol. 4', 'flute archive vol. 4', regex=False)
    result['Items'] = result['Items'].str.replace('flute vol. 5', 'flute archive vol. 5', regex=False)
    result['Items'] = result['Items'].str.replace('flute vol. 6', 'flute archive vol. 6', regex=False)
    result['Items'] = result['Items'].str.replace('flute vol. 7', 'flute archive vol. 7', regex=False)
    result['Items'] = result['Items'].str.replace('flute vols. 1 - 4', 'flute archive vols. 1 - 4', regex=False)
    result['Items'] = result['Items'].str.replace('flute vols. 5-7', 'flute archive vols. 5 - 7', regex=False)
    result['Items'] = result['Items'].str.replace('flute archive vols. 5-7', 'flute archive vols. 5 - 7', regex=False)
    result['Items'] = result['Items'].str.replace('recorder archive vol. 1', 'recorder archive', regex=False)
    result['Items'] = result['Items'].str.replace('sound files for french horn vol. 1', 'sound files for french horn archive vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('sound files for oboe vol. 1', 'sound files for oboe archive vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('sound files for recorder vol. 1', 'sound files for recorder archive', regex=False)
    result['Items'] = result['Items'].str.replace('sound files for saxophone vol. 1', 'sound files for saxophone archive vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('sound files for trumpet vol. 1', 'sound files for trumpet archive vol. 1', regex=False)
    result['Items'] = result['Items'].str.replace('string quartet vol. 1', 'string quartet archive', regex=False)
        
    
    # Ensure data is cleaned of any entries without item names or NaN in 'Items'
    result = result[result['Items'].notna() & (result['Items'].str.strip() != '')]
       
        # Prepare the data
    result['Date'] = pd.to_datetime(result['Date'])  # Ensure the date is in datetime format
    result['count'] = 1  # Initialize a count column for each purchase

    # Filter items that occur more than 0 times
    filtered_result = result.groupby('Items').filter(lambda x: len(x) > 0)

    # Sort data first by item and then by date
    filtered_result.sort_values(by=['Items', 'Date'], inplace=True)

    # Calculate cumulative counts for each item over time
    filtered_result['cumulative_count'] = filtered_result.groupby('Items')['count'].cumsum()

    # Calculate the earliest date for each item
    earliest_dates = filtered_result.groupby('Items')['Date'].min().reset_index()
    earliest_dates.rename(columns={'Date': 'Digital Release Date'}, inplace=True)

    # Merge the release date back into the filtered_result DataFrame
    filtered_result = filtered_result.merge(earliest_dates, on='Items', how='left')
    
    # Convert 'Date' to string format for display purposes only
    filtered_result['Digital Release Date'] = filtered_result['Digital Release Date'].dt.strftime('%Y-%m-%d')
    
    Query = """
    SELECT `Digital Release Date`, Items, MAX(cumulative_count) AS `Quantity Sold`
    FROM filtered_result
    GROUP BY `Digital Release Date`, Items

    """
    filtered_result = pysqldf(Query)
    
    # Sorting and plotting (as per your provided setup)
    summary_table_sorted = filtered_result.sort_values(by='Digital Release Date')
    start_date = summary_table_sorted['Digital Release Date'].min()
    end_date = summary_table_sorted['Digital Release Date'].max()
    
    # Normalize the data to range from 0.2 to 1 to ensure dots are visible but still reflect quantity differences
    opacity_norm = (summary_table_sorted['Quantity Sold'] - summary_table_sorted['Quantity Sold'].min())/(summary_table_sorted['Quantity Sold'].max() - summary_table_sorted['Quantity Sold'].min())
    opacity_scaled = 0.2 + 0.8 * opacity_norm  # Scale opacity between 0.2 and 1
    small_constant = 1e-8

    # Logic for determining what to plot based on checkbox, still declared early
    st.markdown("**Quantity Sold by Item and Release Date:**")
    use_log_scale = st.checkbox("Display Log of Quantity Sold", value=False)

    if use_log_scale:
        summary_table_sorted['Log Quantity Sold'] = np.log(summary_table_sorted['Quantity Sold'] + small_constant)
        y = 'Log Quantity Sold'
        y_title = "Log of Quantity Sold to Date"
    else:
        y = 'Quantity Sold'
        y_title = "Quantity Sold to Date"
    
    # Creating the scatter plot
    fig = px.scatter(summary_table_sorted, 
                     x='Digital Release Date', 
                     y=y, 
                     color='Items', 
                     opacity= 0.6,
                     labels={'Digital Release Date': 'Release Date', 'Quantity Sold': 'Quantity Sold', 'Items': 'Items'},
                     title='')
    
    # Customize the chart appearance
    fig.update_traces(textposition='top center', marker=dict(size=10))
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white', showlegend=True,
        xaxis_tickangle=0, xaxis_title="", yaxis_title=y_title,
        font=dict(family="Times New Roman", size=12, color="black"),
        hovermode='closest'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    # Displaying the result
    st.markdown("**Item Popularity:**")
    st.dataframe(filtered_result)
    
    # Ensure the date is in datetime format for the future
    filtered_result['Date'] = pd.to_datetime(result['Date']) 

# Loyalty & RFM   
if uploaded_file_sales is not None and uploaded_file_customer is not None:
    uploaded_file_customer.seek(0)  # Reset the file pointer to the start of the file every time before reading    
    
    df_payhip = pd.read_csv(uploaded_file_customer, sep = ',')
        
    # Now, calculate Recency, Frequency, and Monetary values

    # Find the global maximum date
    df_payhip['Date'] = pd.to_datetime(df_payhip['Date'])
    max_date = df_payhip['Date'].max()

    # SQL Query to calculate initial values
    query_rfm = """
    SELECT
        Email,
        `Country Name` AS Country,
        MIN(Date) as First_Purchase,
        MAX(Date) as Last_Purchase,
        COUNT(Email) as Frequency, 
        SUM(`Amount Net`) as Monetary
    FROM df_payhip
    GROUP BY Email
    """
    rfm_df = psql.sqldf(query_rfm, locals())

    # Convert 'First_Purchase' and 'Last_Purchase' back to datetime
    rfm_df['First_Purchase'] = pd.to_datetime(rfm_df['First_Purchase'])
    rfm_df['Last_Purchase'] = pd.to_datetime(rfm_df['Last_Purchase'])

    # Calculate Recency and Loyalty in pandas
    rfm_df['Recency'] = (max_date - rfm_df['Last_Purchase']).dt.days  # Days since last purchase
    rfm_df['Loyalty'] = (rfm_df['Last_Purchase'] - rfm_df['First_Purchase']).dt.days  # Duration of engagement in days
    
    
    # Correct SQL query to assign quintiles
    sql_query = """ 
    SELECT *,
        CASE
            WHEN Loyalty <= 10 THEN 1
            WHEN Loyalty <= 60 THEN 2
            WHEN Loyalty <= 365 THEN 3
            ELSE 4
        END AS `Loyalty Score`,
        CASE
            WHEN Recency <= 41 THEN 4
            WHEN Recency <= 70 THEN 3
            WHEN Recency <= 98 THEN 2
            ELSE 1
        END AS `Recency Score`,
        CASE
            WHEN Frequency <= 1 THEN 1
            WHEN Frequency <= 2 THEN 2
            WHEN Frequency <= 3 THEN 3
            ELSE 4
        END AS `Frequency Score`,
        CASE
            WHEN Monetary <= 16 THEN 1
            WHEN Monetary <= 30 THEN 2
            WHEN Monetary <= 46 THEN 3
            ELSE 4
        END AS `Monetary Score`
    FROM rfm_df;
    """

    # Execute the SQL query using pandasql with parameters for thresholds
    rfm = psql.sqldf(sql_query, locals())    

    # Just for backups sake
    df_payhip = df_payhip[['Email', 'Amount Net', 'Items In Cart', 'Payment Type', 'Unsubscribed From Email Updates', 'Country Name', 'Date']]

    df_payhip.rename(columns={'Items In Cart': 'Items'}, inplace=True)
    df_payhip.rename(columns={'Country Name': 'Country'}, inplace=True)
    df_payhip.rename(columns={'Amount Net': 'Customer Value'}, inplace=True)
    df_payhip.rename(columns={'Unsubscribed From Email Updates': 'Email Unsub'}, inplace=True)

    df_payhip['Date'] = pd.to_datetime(df_payhip['Date'])  # Ensure the date is in datetime format

    # Sort data first by item and then by date
    df_payhip.sort_values(by=['Date', 'Email'], inplace=True)
    
    df_payhip['Date'] = pd.to_datetime(df_payhip['Date'])  # Ensure the date is in datetime format

    # Sort data first by item and then by date
    df_payhip.sort_values(by=['Email', 'Date'], inplace=True)

    query_keywords = """
    SELECT
        Email,
        `Email Unsub`,
        `Payment Type`,
        CASE WHEN LOWER(Items) LIKE '%clarinet%' THEN 1 ELSE 0 END AS clarinet,
        CASE WHEN LOWER(Items) LIKE '%oboe%' THEN 1 ELSE 0 END AS oboe,
        CASE WHEN LOWER(Items) LIKE '%flute%' THEN 1 ELSE 0 END AS flute,
        CASE WHEN LOWER(Items) LIKE '%recorder%' THEN 1 ELSE 0 END AS recorder,
        CASE WHEN LOWER(Items) LIKE '%saxophone%' THEN 1 ELSE 0 END AS saxophone,
        CASE WHEN LOWER(Items) LIKE '%brass%' THEN 1 ELSE 0 END AS brass,
        CASE WHEN LOWER(Items) LIKE '%trombone%' THEN 1 ELSE 0 END AS trombone,
        CASE WHEN LOWER(Items) LIKE '%bassoon%' THEN 1 ELSE 0 END AS bassoon,
        CASE WHEN LOWER(Items) LIKE '%trumpet%' THEN 1 ELSE 0 END AS trumpet,
        CASE WHEN LOWER(Items) LIKE '%french_horn%' THEN 1 ELSE 0 END AS 'frenchhorn',
        CASE WHEN LOWER(Items) LIKE '%ww%' THEN 1 ELSE 0 END AS woodwind,
        CASE WHEN LOWER(Items) LIKE '%tuba%' THEN 1 ELSE 0 END AS tuba,
        CASE WHEN LOWER(Items) LIKE '%euphonium%' THEN 1 ELSE 0 END AS euphonium,
        CASE WHEN LOWER(Items) LIKE '%cello%' THEN 1 ELSE 0 END AS cello,
        CASE WHEN LOWER(Items) LIKE '%sound_files%' THEN 1 ELSE 0 END AS 'soundfiles',
        CASE WHEN LOWER(Items) LIKE '%string%' THEN 1 ELSE 0 END AS string,
        CASE WHEN `Email Unsub` LIKE '1' THEN 'unsub' ELSE 'sub' END AS `Email Status`
    FROM df_payhip
    WHERE 
        Email NOT IN ('brahms23@yahoo.com', 'Brahms23@yahoo.com', 'brahms23@gmail.com', 'Brahms23@gmail.com')
        AND Items NOT IN ('AA Clarinet 1 evaluation', 'AA Clarinet 2 evaluation')
    """

    keywords_df = psql.sqldf(query_keywords, locals())

    # Function to create a string of instruments that are 1
    def create_instrument_string(row):
        instruments = []
        for col in row.index[3:]:  # Skip the first three columns (Email, Unsub, and Payment type)
            if row[col] == 1:
                instruments.append(col)
        return ' '.join(instruments)

    # Create the 'instruments' column
    keywords_df['Keywords'] = keywords_df.apply(create_instrument_string, axis=1)

    # Select only Email and instruments columns
    pivoted_result = keywords_df[['Email', 'Keywords', 'Email Status', 'Payment Type']]    

    # Define the SQL query
    Query = """
    SELECT rfm.*, pivoted_result.* 
    FROM rfm
    JOIN pivoted_result ON rfm.Email = pivoted_result.Email
    """

    # Execute the query
    table = psql.sqldf(Query, locals())

    table = table.loc[:, ~table.columns.duplicated()]
    table = table[['Email', 'Loyalty Score', 'Recency Score', 'Frequency Score', 'Monetary Score', 'Keywords', 'Email Status', 'Country','Payment Type']]
    
    table['Keywords'] = table['Keywords'].str.replace(' ', ', ', regex=False)
    
    # Group by 'Email' and join the Keywords
    grouped_table = table.groupby('Email').agg({
        'Email Status': 'last',  # Assuming all entries per email are the same
        'Keywords': lambda x: ', '.join(sorted(set(x))),  # Removes duplicates and sorts
        'Loyalty Score': 'last',  # Adjust according to your data's needs
        'Recency Score': 'last',  # Adjust according to your data's needs
        'Frequency Score': 'last',  # Adjust according to your data's needs
        'Monetary Score': 'last',  # Adjust according to your data's need
        'Country': 'last',  # Assuming all entries per email are the same
        'Payment Type': 'first',  # Assuming all entries per email are the same
    }).reset_index()
    
    # Search bar
    search_query = st.text_input("Enter search term (Email, Country, Keywords, etc.)", "")

    # Filter the DataFrame based on the search query
    if search_query:
        filtered_df = grouped_table[grouped_table.apply(lambda row: row.astype(str).str.contains(search_query, case=False, regex=True).any(), axis=1)]
    else:
        filtered_df = grouped_table

    # Display the DataFrame in Streamlit
    st.markdown("**Customer :rainbow[Loyalty & RFM] Details for :rainbow[Targeted Advertising]:**")
    st.dataframe(filtered_df)
    # REMEMBER Payment Type represents the first type used.
    # Step 1: Calculate the Combined MF Score (Monetary + Frequency Score)
    data = filtered_df
    data['MF Score'] = data['Monetary Score'] + data['Frequency Score']
    
    # Step 2: Find the top 10 most common countries
    top_10_countries = data['Country'].value_counts().nlargest(10).index
    
    # Step 3: Create a new column 'Country_Grouped' where countries outside the top 10 are labeled as 'RoW'
    data['Country_Grouped'] = data['Country'].apply(lambda x: x if x in top_10_countries else 'RoW')
    
    # Step 4: One-hot encode the top 10 countries and 'RoW'
    country_dummies = pd.get_dummies(data['Country_Grouped'], drop_first=True)  
    
    # Step 5: Drop the original 'Country' and 'Country_Grouped' columns (if no longer needed)
    data = data.drop(columns=['Country', 'Country_Grouped'])
    
    # Step 6: Split the Keywords column into individual keywords
    # Ensure consistent splitting by replacing any spaces after commas
    data['Keywords'] = data['Keywords'].str.replace(", ", ",")
    keywords_split = data['Keywords'].str.get_dummies(sep=',')

    # Manually drop the first column to avoid multicollinearity
    if not keywords_split.empty:
        keywords_split = keywords_split.iloc[:, 1:]
    
    # Step 7: Perform one-hot encoding for other categorical variables (Email Status and Payment Type)
    other_dummies = pd.get_dummies(data[['Email Status', 'Payment Type']], drop_first=True)
    
    # Step 8: Combine the keyword dummies, country dummies, and other categorical dummies
    encoded_data = pd.concat([data, country_dummies, keywords_split, other_dummies], axis=1)

    # Convert all one-hot encoded columns to integer type to ensure uniformity
    # This includes dummies from country, keywords, and other categorical variables
    one_hot_encoded_columns = list(country_dummies.columns) + list(keywords_split.columns) + list(other_dummies.columns)
    for column in one_hot_encoded_columns:
        encoded_data[column] = encoded_data[column].astype(int)
    
    # Step 9: Drop unnecessary columns like 'Email Status', 'Payment Type', 'Keywords'
    encoded_data = encoded_data.drop(columns=['Email Status', 'Payment Type', 'Keywords'])
    # Create a new column 'binary_var' with 1 if 'original_var' > 5, else 0
    encoded_data['VIP'] = (encoded_data['Monetary Score'] > 3).astype(int)

    # Function to detect readable emails
    def check_readable_emails(df):
        email_domains = ["@gmail", "@yahoo", "@hotmail", "@icloud", "@aol"]
        
        # Checking if any email in the "Email" column contains one of the domain strings
        if encoded_data['Email'].str.contains('|'.join(email_domains)).any():
            return True
        else:
            return False

    # Checking if the "Email" column exists
    if 'Email' in encoded_data.columns:
        # Check if the column contains readable emails
        if check_readable_emails(encoded_data):
            # If readable emails are detected, create dummy columns
            encoded_data['Email_Domain'] = encoded_data['Email'].apply(lambda x: '@gmail' if '@gmail' in x else
                                                              '@yahoo' if '@yahoo' in x else
                                                              '@hotmail' if '@hotmail' in x else
                                                              '@icloud' if '@icloud' in x else
                                                              '@aol' if '@aol' in x else '@other')
            # Create dummy variables and ensure they're integers (1s and 0s)
            df_dummies = pd.get_dummies(encoded_data['Email_Domain'], prefix='Domain').astype(int)
            # Concatenate the dummy columns to the original dataframe
            encoded_data = pd.concat([encoded_data, df_dummies], axis=1)
            # Drop the original 'Email' column
            encoded_data = encoded_data.drop('Email_Domain', axis=1)

            # 1. Set y (the dependent variable) as the 'MF Score' column
            y = encoded_data['VIP']
            # 2. Set X (the independent variables) as all columns except 'MF Score'
            X = encoded_data.drop(columns=['VIP','MF Score', 'Monetary Score', 'Recency Score', 'Frequency Score', 'Email', 'Loyalty Score', 'Domain_@gmail'])
            # Add a constant term to the regression
            X = sm.add_constant(X)
            # Optionally convert X and y to numpy arrays if required by the model
            # X = X.values  # Converts X to a NumPy array
            # y = y.values  # Converts y to a NumPy array
        else:
                # 1. Set y (the dependent variable) as the 'MF Score' column
            y = encoded_data['VIP']
            # 2. Set X (the independent variables) as all columns except 'MF Score'
            X = encoded_data.drop(columns=['VIP','MF Score', 'Monetary Score', 'Recency Score', 'Frequency Score', 'Email', 'Loyalty Score'])
            # Add a constant term to the regression
            X = sm.add_constant(X)
            # Optionally convert X and y to numpy arrays if required by the model
            # X = X.values  # Converts X to a NumPy array
            # y = y.values  # Converts y to a NumPy array

    model = sm.Logit(y, X)
    result = model.fit()
    # Calculate the marginal effects
    marginal_effects = result.get_margeff()
    marginal_effects_df = marginal_effects.summary_frame()
    # DO NOT Display Regression Results. DONT MAKE CUELLAR MAD !
    # Filter significant coefficients (e.g., p-value < 0.05)
    significant_margeff = marginal_effects_df[marginal_effects_df['Pr(>|z|)'] < 0.05]
    # Sort the DataFrame by the lower bound of the coefficient
    significant_margeff = significant_margeff.sort_values('dy/dx', ascending=True)

    # Apply a style template that's close to Streamlit's default style
    plt.style.use('ggplot')
    # Error bars calculated from confidence intervals
    error_y = [significant_margeff['dy/dx'] - significant_margeff['Conf. Int. Low'],
                  significant_margeff['Cont. Int. Hi.'] - significant_margeff['dy/dx']]
    

    # Create the error bar graph
    fig = go.Figure(data=go.Scatter(
        x=significant_margeff.index,
        y=significant_margeff['dy/dx'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=error_y[1],
            arrayminus=error_y[0],
            color='lightblue',
            thickness=1.75,
            width=3,
        ),
        mode='markers',
        marker=dict(size=10, color='blue', opacity=0.4)  # Using a simple color for demonstration
    ))
    
    # Customize the layout to match the previous style
    fig.update_layout(
        title="What Makes a Top Quartile Spender? Let's Use Logistic Regression to Find Out",
        xaxis_title="Statistically Significant at a 0.05 level",
        yaxis_title="Change in % Chance of a Whale",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(tickangle=45),
        font=dict(family="Times New Roman", size=12, color="black"),
        hovermode='closest',
        margin=dict(l=40, r=40, t=40, b=80)  # Adjust margins as needed
    )
    
    # Base cases note at the bottom
    fig.add_annotation(
        x=0,
        y=1.05,
        xref='paper',
        yref='paper',
        text="Binary Base Cases: Country: Australia, Instrument: Bassoon, Payment type: Free, Email: Gmail",
        showarrow=False,
        font=dict(size=12),
        align='center'
    )
    
    # Display the plot in Streamlit
    st.markdown(" ")
    st.plotly_chart(fig, use_container_width=True)
    
    
# find me f key
# Sales grouped by Email Unsub & Payment Type    
if uploaded_file_sales is not None and uploaded_file_customer is not None:
    uploaded_file_customer.seek(0)  # Reset the file pointer to the start of the file every time before reading       
    uploaded_file_sales.seek(0)
    Payhip = pd.read_csv(uploaded_file_customer, sep = ',')
    
    # Function to use pandasql
    pysqldf = lambda q: sqldf(q, globals())    
    sqlQuery = """
    SELECT *
    FROM Payhip
    WHERE Email NOT IN ('brahms23@yahoo.com', 'Brahms23@yahoo.com', 'brahms23@gmail.com', 'Brahms23@gmail.com')
    AND `Items In Cart` NOT IN ('AA Clarinet 1 evaluation', 'AA Clarinet 2 evaluation')
    """
    Result = pysqldf(sqlQuery)
    
    # Naming
    # Rename the 'Amount Net' column to 'Sales'
    Result.rename(columns={'Items In Cart': 'Items'}, inplace=True)
    Result.rename(columns={'Country Name': 'Country'}, inplace=True)
    Result.rename(columns={'Unsubscribed From Email Updates': 'Email Unsub'}, inplace=True)
    
    Sqlquery = """
    Select SUM(`Amount Net`) AS amount_net, AVG(`Amount Net`) AS amount_net_avg,
    `Email Unsub`, `Payment Type`,
    CASE WHEN `Email Unsub` LIKE '1' THEN 'unsub' ELSE 'sub' END AS `Email Status`
    FROM Result
    GROUP BY `Payment Type`, `Email Unsub`
    ORDER BY `Payment Type` ASC, `Email Status` ASC
    """
    
    another_df = pysqldf(Sqlquery)
    
    another_df = another_df[["amount_net", 'amount_net_avg', 'Email Status', 'Payment Type']]
    
    SqlQuery = """
    Select 
    SUM(`Amount Net`) AS `Net Revenue`, 
    AVG(`Amount Net`) AS `Avg Net Revenue`, 
    `Payment Type`, 
    COUNT(`Payment Type`) AS `Payment Count`
    FROM Result
    GROUP BY `Payment Type`
    ORDER BY `Payment Type` ASC
    """
    
    Another_df = pysqldf(SqlQuery)
    
    Another_df = Another_df[["Net Revenue", 'Avg Net Revenue', 'Payment Type', 'Payment Count']]
    
    sqlquery = """
    SELECT 
    CASE `Email Unsub`
        WHEN 0 THEN 'sub'
        WHEN 1 THEN 'unsub'
        ELSE `Email Unsub`
    END AS `Subscription Status`,
    COUNT(`Email Unsub`) AS `Count`
FROM Result
GROUP BY 
    CASE `Email Unsub`
        WHEN 0 THEN 'sub'
        WHEN 1 THEN 'unsub'
        ELSE `Email Unsub`
    END
    """
    unsub_df = pysqldf(sqlquery)
    
    
    # Bar Graph
    def plot_data(df):
        # Create figure with secondary y-axis
        Fig = go.Figure()

        # Add traces
        Fig.add_trace(
            go.Bar(
                x=df['Payment Type'] + ' ' + df['Email Status'],
                y=df['amount_net'],
                name='Sum of Digital Sales',
                marker_color='indigo'
            )
        )

        Fig.add_trace(
            go.Scatter(
                x=df['Payment Type'] + ' ' + df['Email Status'],
                y=df['amount_net_avg'],
                name='Average Digital Sale',
                yaxis='y2',
                marker_color='green'
            )
        )

        # Add titles and labels
        Fig.update_layout(
            title='Digital Sales grouped by Email Status & Payment Type:',
            xaxis_title='',
            yaxis_title='Sum of Digital Sales',
            yaxis2=dict(
                title='Average Digital Sale',
                overlaying='y',
                side='right'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            font=dict(
                family="Times New Roman",
                size=12,
                color="black"
            )
        )

        return Fig
    
    # Display the plotly in Streamlit
    st.markdown("**Payment Type:**")
    st.dataframe(Another_df, hide_index = True)
    st.markdown("**Email Count:**")
    st.dataframe(unsub_df, hide_index = True)
    Fig = plot_data(another_df)
    st.plotly_chart(Fig, use_container_width=True)
    st.markdown(":green[Paypal email subscribers spend **more** per purchase than those that unsub, while Stripe subs spend **less** per purchase than those that unsub. Despite this, Stripe users tend to spend **more** per purchase than Paypal users.]")
    st.markdown(":violet[We can see that purchases with the **Free** payment type have not spent a dime, conversely it also has the highest probabilty of a VIP in the Payment Type category. This is becasue the payment type variable corresponds **only** to the most recent payment type used by the customer. Free means they recieved a free product from customer service as their most recent order.]")
    st.markdown(":blue[**From this we glean that those reaching out to customer service are already the biggest spenders**]")
    st.markdown(" ")


# Spending by Email Domain 
if uploaded_file_sales is not None and uploaded_file_customer is not None:
    uploaded_file_customer.seek(0)  # Reset the file pointer to the start of the file every time before reading       
    uploaded_file_sales.seek(0)    
    
    hip_df = pd.read_csv(uploaded_file_customer, sep=',')
    hip_df = hip_df[["Amount Net", 'Email']]
    if check_readable_emails(hip_df):
                hip_df['Email_Domain'] = hip_df['Email'].apply(lambda x: '@gmail' if '@gmail' in x else
                                                                  '@yahoo' if '@yahoo' in x else
                                                                  '@hotmail' if '@hotmail' in x else
                                                                  '@icloud' if '@icloud' in x else
                                                                  '@aol' if '@aol' in x else '@other')
                # Create dummy variables and ensure they're integers (1s and 0s)
                df_dummies = pd.get_dummies(hip_df['Email_Domain'], prefix='Domain').astype(int)
                # Concatenate the dummy columns to the original dataframe
                spendbyemail_df = pd.concat([hip_df, df_dummies], axis=1)
                # Drop the original 'Email' column and everythings else
                spendbyemail_df = spendbyemail_df.drop(['Email_Domain', 'Email'], axis=1)

            

# Country Metrics
if uploaded_file_sales is not None and uploaded_file_customer is not None:
    uploaded_file_customer.seek(0)  # Reset the file pointer to the start of the file every time before reading       
    uploaded_file_sales.seek(0)    
    
    hip_df = pd.read_csv(uploaded_file_customer, sep=',')
    hip_df = hip_df[["Amount Net", 'Country Name']]
    
    # Ensure 'Country Name' has no None values
    hip_df.dropna(subset=['Country Name'], inplace=True)
    hip_df['Country Name'] = hip_df['Country Name'].astype(str)
    
    country_query = """
        SELECT 
        SUM(`Amount Net`) AS `Net Revenue`,
        `Country Name` AS `Country`,
        COUNT(`Country Name`) AS Count,
        AVG(`Amount Net`) AS `Avg Net Revenue`
        FROM hip_df
        GROUP BY `Country Name`
    """
    country_df = pysqldf(country_query)
    
  # User selection for metrics to display
    st.markdown("**Country Metrics Comparison:**")
    st.dataframe(country_df.describe())
    show_count = st.checkbox('Show Transaction Count', True)
    show_avg_net_revenue = st.checkbox('Show Average Net Revenue', True)
    show_net_revenue = st.checkbox('Show Net Revenue', True)

    plt.style.use('seaborn-v0_8-whitegrid')
   
    if show_avg_net_revenue or show_count or show_net_revenue:
        # Sorting
        sort_metric = 'Avg Net Revenue' if show_avg_net_revenue else 'Count' if show_count else 'Net Revenue'
        country_df.sort_values(by=sort_metric, ascending=False, inplace=True)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color_net_rev = 'tab:orange'
        color_count = 'indigo'
        color_avg_net_rev = 'tab:cyan'
        
        # Spacing between bars
        bar_width = 0.7
        bar_width_count = 0.35
        index = range(len(country_df))

        if show_avg_net_revenue:
            ax1.bar(index, country_df['Avg Net Revenue'], color=color_avg_net_rev, width=bar_width, label='Avg Net Revenue', zorder=3)
            ax1.set_ylabel('', color='tab:grey')
            ax1.tick_params(axis='y', labelcolor='tab:grey', length = 0)

        if show_net_revenue:
            ax1.set_ylabel('', color='tab:grey')
            ax1.tick_params(axis='y', labelcolor='tab:grey', length = 0)
            ax2 = ax1.twinx()
            ax2.plot(country_df['Country'], country_df['Net Revenue'], color=color_net_rev, marker='o', linestyle='-', label='Net Revenue', linewidth=2, markeredgewidth=0, zorder=3)
            ax2.set_ylabel('', color='tab:grey')
            ax2.tick_params(axis='y', labelcolor='tab:grey', length = 0)
            ax2.set_frame_on(False)

        if show_count:
            ax1.bar([p + bar_width_count for p in index], country_df['Count'], color=color_count, width=bar_width_count, alpha=1, label='Count of Purchases', zorder=3)
            ax1.set_ylabel('', color='tab:grey')
            ax1.tick_params(axis='y', labelcolor='tab:grey', length = 0)

            
        ax1.grid(True, which='major', axis='y', linestyle='-', linewidth=0.57, color='lightgrey', zorder=0)
        ax1.grid(False, which='major', axis='x')  # Explicitly disable x-axis grid lines
        ax1.set_xlabel('')
        ax1.set_title('Country Metrics')
        ax1.set_xticks([p + bar_width / 2 for p in index])
        ax1.set_xticklabels(country_df['Country'], rotation=90, ha="right", fontname='Arial', color='tab:grey', fontsize =10)
        fig.tight_layout()

        # Remove spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        
        # Adding a legend that displays all metrics
        handles, labels = [], []
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        plt.legend(handles, labels, loc='upper right', frameon=False)
        st.pyplot(fig)
    else:
        st.write("Please select at least one metric to display.")
        
    
        
        
# Country Keyword Metrics
if uploaded_file_sales is not None and uploaded_file_customer is not None:
    uploaded_file_customer.seek(0)  # Reset the file pointer to the start of the file every time before reading       
    uploaded_file_sales.seek(0)  

    df_payhip = pd.read_csv(uploaded_file_customer, sep = ',')
    
    # Sort data first by item and then by date
    df_payhip.sort_values(by=['Email', 'Date'], inplace=True)
    
    # Rename the 'Sales' column to 'Physical Sales'
    df_payhip.rename(columns={'Country Name': 'Country'}, inplace=True)
    df_payhip.rename(columns={'Items In Cart': 'Items'}, inplace=True)
    df_payhip.rename(columns={'Unsubscribed From Email Updates': 'Email Unsub'}, inplace=True)

    query_keywords = """
    SELECT
        `Country`,
        `Email Unsub`,
        `Payment Type`,
        CASE WHEN LOWER(Items) LIKE '%clarinet%' THEN 1 ELSE 0 END AS clarinet,
        CASE WHEN LOWER(Items) LIKE '%oboe%' THEN 1 ELSE 0 END AS oboe,
        CASE WHEN LOWER(Items) LIKE '%flute%' THEN 1 ELSE 0 END AS flute,
        CASE WHEN LOWER(Items) LIKE '%recorder%' THEN 1 ELSE 0 END AS recorder,
        CASE WHEN LOWER(Items) LIKE '%saxophone%' THEN 1 ELSE 0 END AS saxophone,
        CASE WHEN LOWER(Items) LIKE '%brass%' THEN 1 ELSE 0 END AS brass,
        CASE WHEN LOWER(Items) LIKE '%trombone%' THEN 1 ELSE 0 END AS trombone,
        CASE WHEN LOWER(Items) LIKE '%bassoon%' THEN 1 ELSE 0 END AS bassoon,
        CASE WHEN LOWER(Items) LIKE '%trumpet%' THEN 1 ELSE 0 END AS trumpet,
        CASE WHEN LOWER(Items) LIKE '%french_horn%' THEN 1 ELSE 0 END AS 'frenchhorn',
        CASE WHEN LOWER(Items) LIKE '%ww%' THEN 1 ELSE 0 END AS woodwind,
        CASE WHEN LOWER(Items) LIKE '%tuba%' THEN 1 ELSE 0 END AS tuba,
        CASE WHEN LOWER(Items) LIKE '%euphonium%' THEN 1 ELSE 0 END AS euphonium,
        CASE WHEN LOWER(Items) LIKE '%cello%' THEN 1 ELSE 0 END AS cello,
        CASE WHEN LOWER(Items) LIKE '%sound_files%' THEN 1 ELSE 0 END AS 'soundfiles',
        CASE WHEN LOWER(Items) LIKE '%string%' THEN 1 ELSE 0 END AS string,
        CASE WHEN `Email Unsub` LIKE '1' THEN 'unsub' ELSE 'sub' END AS `Email Status`,
        CASE WHEN `Payment Type` LIKE 'stripe' THEN '1' ELSE '0' END AS `Stripe`
    FROM df_payhip
    WHERE 
        Email NOT IN ('brahms23@yahoo.com', 'Brahms23@yahoo.com', 'brahms23@gmail.com', 'Brahms23@gmail.com')
        OR Items NOT IN ('AA Clarinet 1 evaluation', 'AA Clarinet 2 evaluation')
    """

    keywords_df = psql.sqldf(query_keywords, locals())
    
    
    query_keywords_summary = """
SELECT
    Country,
    SUM(clarinet) AS Sum_Clarinet,
    AVG(clarinet) AS Avg_Clarinet,
    SUM(oboe) AS Sum_Oboe,
    AVG(oboe) AS Avg_Oboe,
    SUM(flute) AS Sum_Flute,
    AVG(flute) AS Avg_Flute,
    SUM(recorder) AS Sum_Recorder,
    AVG(recorder) AS Avg_Recorder,
    SUM(saxophone) AS Sum_Saxophone,
    AVG(saxophone) AS Avg_Saxophone,
    SUM(brass) AS Sum_Brass,
    AVG(brass) AS Avg_Brass,
    SUM(trombone) AS Sum_Trombone,
    AVG(trombone) AS Avg_Trombone,
    SUM(bassoon) AS Sum_Bassoon,
    AVG(bassoon) AS Avg_Bassoon,
    SUM(trumpet) AS Sum_Trumpet,
    AVG(trumpet) AS Avg_Trumpet,
    SUM(frenchhorn) AS Sum_FrenchHorn,
    AVG(frenchhorn) AS Avg_FrenchHorn,
    SUM(woodwind) AS Sum_Woodwind,
    AVG(woodwind) AS Avg_Woodwind,
    SUM(tuba) AS Sum_Tuba,
    AVG(tuba) AS Avg_Tuba,
    SUM(euphonium) AS Sum_Euphonium,
    AVG(euphonium) AS Avg_Euphonium,
    SUM(cello) AS Sum_Cello,
    AVG(cello) AS Avg_Cello,
    SUM(soundfiles) AS Sum_SoundFiles,
    AVG(soundfiles) AS Avg_SoundFiles,
    SUM(string) AS Sum_String,
    AVG(string) AS Avg_String
FROM keywords_df
GROUP BY Country
"""
    keywords_summary_df = psql.sqldf(query_keywords_summary, locals())
    
    # Display 
    # Set 'Country' as the index and plot all relevant 'SUM_' and 'AVG_' prefixed columns
    # Create a checkbox in the sidebar to toggle between sums and averages
    st.markdown("**Global Item Popularity:**")
    show_averages = st.checkbox('Show Averages/Percentages', value=False)

    # Set 'Country' as the index
    keywords_summary_df.set_index('Country', inplace=True)

    if show_averages:
        # If the checkbox is checked, display the averages
        columns_to_display = [
        'Avg_Clarinet', 'Avg_Oboe', 'Avg_Trumpet', 'Avg_FrenchHorn', 'Avg_Flute', 
        'Avg_Saxophone', 'Avg_Bassoon', 'Avg_Cello', 'Avg_Woodwind', 
        'Avg_Tuba', 'Avg_Euphonium', 'Avg_SoundFiles', 
        'Avg_String', 'Avg_Brass', 'Avg_Trombone', 'Avg_Recorder']
    else:
        # If the checkbox is not checked, display the sums
        columns_to_display = [
        'Sum_Clarinet', 'Sum_Oboe', 'Sum_Trumpet', 'Sum_FrenchHorn', 'Sum_Flute', 
        'Sum_Saxophone', 'Sum_Bassoon', 'Sum_Cello', 'Sum_Woodwind', 
        'Sum_Tuba', 'Sum_Euphonium', 'Sum_SoundFiles', 
        'Sum_String', 'Sum_Brass', 'Sum_Trombone', 'Sum_Recorder']

    # Display the bar chart with the selected columns
    st.bar_chart(keywords_summary_df[columns_to_display])
    
    QQ = """
        SELECT Email, Date, 
        strftime('%Y', Date) as Year,
        strftime('%m', Date) as Month,
            CASE WHEN LOWER(Items) LIKE '%clarinet%' THEN 1 ELSE 0 END AS clarinet,
            CASE WHEN LOWER(Items) LIKE '%oboe%' THEN 1 ELSE 0 END AS oboe,
            CASE WHEN LOWER(Items) LIKE '%flute%' THEN 1 ELSE 0 END AS flute,
            CASE WHEN LOWER(Items) LIKE '%recorder%' THEN 1 ELSE 0 END AS recorder,
            CASE WHEN LOWER(Items) LIKE '%saxophone%' THEN 1 ELSE 0 END AS saxophone,
            CASE WHEN LOWER(Items) LIKE '%brass%' THEN 1 ELSE 0 END AS brass,
            CASE WHEN LOWER(Items) LIKE '%trombone%' THEN 1 ELSE 0 END AS trombone,
            CASE WHEN LOWER(Items) LIKE '%bassoon%' THEN 1 ELSE 0 END AS bassoon,
            CASE WHEN LOWER(Items) LIKE '%trumpet%' THEN 1 ELSE 0 END AS trumpet,
            CASE WHEN LOWER(Items) LIKE '%french_horn%' THEN 1 ELSE 0 END AS 'frenchhorn',
            CASE WHEN LOWER(Items) LIKE '%ww%' THEN 1 ELSE 0 END AS woodwind,
            CASE WHEN LOWER(Items) LIKE '%tuba%' THEN 1 ELSE 0 END AS tuba,
            CASE WHEN LOWER(Items) LIKE '%euphonium%' THEN 1 ELSE 0 END AS euphonium,
            CASE WHEN LOWER(Items) LIKE '%cello%' THEN 1 ELSE 0 END AS cello,
            CASE WHEN LOWER(Items) LIKE '%sound_files%' THEN 1 ELSE 0 END AS 'soundfiles',
            CASE WHEN LOWER(Items) LIKE '%string%' THEN 1 ELSE 0 END AS string
        FROM df_payhip
        WHERE 
        Email NOT IN ('brahms23@yahoo.com', 'Brahms23@yahoo.com', 'brahms23@gmail.com', 'Brahms23@gmail.com')
        OR Items NOT IN ('AA Clarinet 1 evaluation', 'AA Clarinet 2 evaluation')
        GROUP BY Date
    """
    QQ_df = psql.sqldf(QQ, locals())
    
    QQQ = """
    SELECT 
    Date,
    SUM(clarinet) AS Sum_Clarinet, AVG(clarinet) AS Avg_Clarinet,
    SUM(oboe) AS Sum_Oboe, AVG(oboe) AS Avg_Oboe,
    SUM(flute) AS Sum_Flute, AVG(flute) AS Avg_Flute,
    SUM(recorder) AS Sum_Recorder, AVG(recorder) AS Avg_Recorder,
    SUM(saxophone) AS Sum_Saxophone, AVG(saxophone) AS Avg_Saxophone,
    SUM(brass) AS Sum_Brass, AVG(brass) AS Avg_Brass,
    SUM(trombone) AS Sum_Trombone, AVG(trombone) AS Avg_Trombone,
    SUM(bassoon) AS Sum_Bassoon, AVG(bassoon) AS Avg_Bassoon,
    SUM(trumpet) AS Sum_Trumpet, AVG(trumpet) AS Avg_Trumpet,
    SUM(frenchhorn) AS Sum_FrenchHorn, AVG(frenchhorn) AS Avg_FrenchHorn,
    SUM(woodwind) AS Sum_Woodwind, AVG(woodwind) AS Avg_Woodwind,
    SUM(tuba) AS Sum_Tuba, AVG(tuba) AS Avg_Tuba,
    SUM(euphonium) AS Sum_Euphonium, AVG(euphonium) AS Avg_Euphonium,
    SUM(cello) AS Sum_Cello, AVG(cello) AS Avg_Cello,
    SUM(soundfiles) AS Sum_SoundFiles, AVG(soundfiles) AS Avg_SoundFiles,
    SUM(string) AS Sum_String, AVG(string) AS Avg_String
FROM QQ_df
GROUP BY Year, Month
ORDER BY Date
    """
    QQQ_df = psql.sqldf(QQQ, locals())
        
    # Set 'Country' as the index
    QQQ_df.set_index('Date', inplace=True)

    if show_averages:
        # If the checkbox is checked, display the averages
        columns_to_display = [
        'Avg_Clarinet', 'Avg_Oboe', 'Avg_Trumpet', 'Avg_FrenchHorn', 'Avg_Flute', 
        'Avg_Saxophone', 'Avg_Bassoon', 'Avg_Cello', 'Avg_Woodwind', 
        'Avg_Tuba', 'Avg_Euphonium', 'Avg_SoundFiles', 
        'Avg_String', 'Avg_Brass', 'Avg_Trombone', 'Avg_Recorder']
    else:
        # If the checkbox is not checked, display the sums
        columns_to_display = [
        'Sum_Clarinet', 'Sum_Oboe', 'Sum_Trumpet', 'Sum_FrenchHorn', 'Sum_Flute', 
        'Sum_Saxophone', 'Sum_Bassoon', 'Sum_Cello', 'Sum_Woodwind', 
        'Sum_Tuba', 'Sum_Euphonium', 'Sum_SoundFiles', 
        'Sum_String', 'Sum_Brass', 'Sum_Trombone', 'Sum_Recorder']

    # Display the bar chart with the selected columns
    st.markdown("**Monthly Item Popularity Over Time:**")
    st.dataframe(QQQ_df[columns_to_display].describe())
    st.markdown(":blue[Product Demand remains relatively constant over time.]")
    st.bar_chart(QQQ_df[columns_to_display])
    
    Q = """
    SELECT 
            Country,
            SUM(`Email Unsub`) AS Sum_Email_Unsub,
            SUM(`Stripe`) AS Sum_Stripe,
            AVG(`Email Unsub`) AS Avg_Email_Unsub,
            AVG(`Stripe`) AS Avg_Stripe
            FROM keywords_df
            GROUP BY Country
       """
    Q_df = psql.sqldf(Q, locals())
    
    
    # Display
    st.markdown("**Global Email and Payment Preference:**")

    # Set 'Country' as the index
    Q_df.set_index('Country', inplace=True)

    if show_averages:
        # If the checkbox is checked, display the averages
        columns_to_display = [
            'Avg_Email_Unsub', 'Avg_Stripe'
        ]
    else:
        # If the checkbox is not checked, display the sums
        columns_to_display = [
            'Sum_Email_Unsub', 'Sum_Stripe'
        ]

    # Display the bar chart with the selected columns
    st.bar_chart(Q_df[columns_to_display], stack=False)
    
# What makes a Email Unsubscriber
if uploaded_file_sales is not None and uploaded_file_customer is not None:
    uploaded_file_customer.seek(0)  # Reset the file pointer to the start of the file every time before reading       
    uploaded_file_sales.seek(0) 

    # Checking if the "Email" column exists
    if 'Email' in encoded_data.columns:
        # Check if the column contains readable emails
        if check_readable_emails(encoded_data):
            # 1. Set y (the dependent variable) as the 'MF Score' column
            y = encoded_data['Email Status_unsub']
            # 2. Set X (the independent variables) as all columns except 'MF Score'
            X = encoded_data.drop(columns=['VIP','MF Score', 'Email', 'Email Status_unsub', 'Domain_@gmail'])
            # Add a constant term to the regression
            X = sm.add_constant(X)
            # Optionally convert X and y to numpy arrays if required by the model
            # X = X.values  # Converts X to a NumPy array
            # y = y.values  # Converts y to a NumPy array
        else:
             # 1. Set y (the dependent variable) as the 'MF Score' column
            y = encoded_data['Email Status_unsub']
            # 2. Set X (the independent variables) as all columns except 'MF Score'
            X = encoded_data.drop(columns=['VIP','MF Score', 'Email', 'Email Status_unsub'])
            # Add a constant term to the regression
            X = sm.add_constant(X)
            # Optionally convert X and y to numpy arrays if required by the model
            # X = X.values  # Converts X to a NumPy array
            # y = y.values  # Converts y to a NumPy array

    model = sm.Logit(y, X)
    result = model.fit()
    # Calculate the marginal effects
    marginal_effects = result.get_margeff()
    marginal_effects_df = marginal_effects.summary_frame()
    # DO NOT Display Regression Results. DONT MAKE CUELLAR MAD !
    # Filter significant coefficients (e.g., p-value < 0.05)
    significant_margeff = marginal_effects_df[marginal_effects_df['Pr(>|z|)'] < 0.05]
    # Sort the DataFrame by the lower bound of the coefficient
    significant_margeff = significant_margeff.sort_values('dy/dx', ascending=True)

    # Apply a style template that's close to Streamlit's default style
    plt.style.use('ggplot')
    # Error bars calculated from confidence intervals
    error_y = [significant_margeff['dy/dx'] - significant_margeff['Conf. Int. Low'],
                  significant_margeff['Cont. Int. Hi.'] - significant_margeff['dy/dx']]
    
    # Create the error bar graph
    fig = go.Figure(data=go.Scatter(
        x=significant_margeff.index,
        y=significant_margeff['dy/dx'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=error_y[1],
            arrayminus=error_y[0],
            color='lightblue',
            thickness=1.75,
            width=3,
        ),
        mode='markers',
        marker=dict(size=10, color='blue', opacity=0.4)  # Using a simple color for demonstration
    ))
    
    # Customize the layout to match the previous style
    fig.update_layout(
        title="Showing What Makes an Email Unsubscriber with Logistic Regression",
        xaxis_title="Statistically Significant at a 0.05 level",
        yaxis_title="Change in % Chance of Customer Unsubscribing",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(tickangle=45),
        font=dict(family="Times New Roman", size=12, color="black"),
        hovermode='closest',
        margin=dict(l=40, r=40, t=40, b=80)  # Adjust margins as needed
    )
    
    # Base cases note at the bottom
    fig.add_annotation(
        x=0,
        y=1.05,
        xref='paper',
        yref='paper',
        text="Binary Base Cases: Country: Australia, Instrument: Bassoon, Payment type: Free, Email: Gmail",
        showarrow=False,
        font=dict(size=12),
        align='center'
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(":blue[Europeans are unsubscribing, but their favorite products are keeping them here. Marketing strategies are being re-designed to boost European engagement.]")

    
        

    
