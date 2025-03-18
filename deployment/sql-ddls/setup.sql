-- 1. forecasts 테이블 생성
CREATE TABLE IF NOT EXISTS public.forecasts (
    id varchar(50) PRIMARY KEY,
    description text,
    created_at timestamptz DEFAULT CURRENT_TIMESTAMP
);

-- 기본 forecast 데이터 추가
INSERT INTO public.forecasts (id, description) 
VALUES ('cbo', 'Congressional Budget Office Forecasts')
ON CONFLICT (id) DO NOTHING;

-- 2. forecast_variables 테이블 생성
CREATE TABLE IF NOT EXISTS public.forecast_variables (
    varname varchar(50) PRIMARY KEY,
    description text,
    created_at timestamptz DEFAULT CURRENT_TIMESTAMP
);

-- 기본 변수들 추가
INSERT INTO public.forecast_variables (varname, description) VALUES
    ('ngdp', 'Gross Domestic Product (GDP)'),
    ('gdp', 'Real GDP'),
    ('pce', 'Personal Consumption Expenditures'),
    ('pdi', 'Gross Private Domestic Investment'),
    ('pdin', 'Nonresidential fixed investment'),
    ('pdir', 'Residential fixed investment'),
    ('cbi', 'Change in private inventories'),
    ('govt', 'Government Consumption Expenditures and Gross Investment'),
    ('govtf', 'Federal'),
    ('govts', 'State and local'),
    ('nx', 'Net Exports of Goods and Services'),
    ('ex', 'Exports'),
    ('im', 'Imports'),
    ('cpi', 'Consumer Price Index, All Urban Consumers (CPI-U)'),
    ('pcepi', 'Price Index, Personal Consumption Expenditures (PCE)'),
    ('oil', 'Price of Crude Oil, West Texas Intermediate (WTI)'),
    ('ffr', 'Federal Funds Rate'),
    ('t03m', '3-Month Treasury Bill'),
    ('t10y', '10-Year Treasury Note'),
    ('unemp', 'Unemployment Rate, Civilian, 16 Years or Older'),
    ('lfpr', 'Labor Force Participation Rate, 16 Years or Older')
ON CONFLICT (varname) DO NOTHING;

-- 3. forecast_values_v2 테이블 생성
CREATE TABLE IF NOT EXISTS public.forecast_values_v2 (
    mdate date not null,
    forecast varchar(50) NOT NULL,
    vdate date NOT NULL,
    form varchar(50) NOT NULL,
    freq bpchar(1) NOT NULL,
    varname varchar(50) NOT NULL,
    "date" date NOT NULL,
    value numeric(20, 4) NOT NULL,
    created_at timestamptz NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT forecast_values_v2_pkey PRIMARY KEY (mdate, forecast, vdate, form, freq, varname, date),
    CONSTRAINT forecast_values_v2_forecast_fk FOREIGN KEY (forecast) REFERENCES public.forecasts(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT forecast_values_v2_varname_fk FOREIGN KEY (varname) REFERENCES public.forecast_variables(varname) ON DELETE CASCADE ON UPDATE CASCADE
);

-- TimescaleDB hypertable 생성 (TimescaleDB가 설치된 경우에만 실행)
-- SELECT create_hypertable('forecast_values_v2', 'mdate', if_not_exists => TRUE);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS forecast_values_v2_idx ON forecast_values_v2 (forecast, vdate, form, freq, varname, date); 