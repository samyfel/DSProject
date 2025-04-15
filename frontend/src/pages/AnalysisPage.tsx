import React from 'react';
import { Typography, Paper, Box, Container, Tabs, Tab, Card, CardContent } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { styled } from '@mui/material/styles';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  marginBottom: theme.spacing(4),
  background: 'linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%)',
  borderRadius: '16px',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
}));

const ResultCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  borderRadius: '16px',
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
  transition: 'transform 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-5px)',
  },
}));

const ChartContainer = styled(Box)(({ theme }) => ({
  height: '400px',
  padding: theme.spacing(2),
  backgroundColor: 'white',
  borderRadius: '12px',
  boxShadow: '0 2px 12px rgba(0, 0, 0, 0.05)',
  marginBottom: theme.spacing(4),
}));

const ImageContainer = styled(Box)(({ theme }) => ({
  width: '100%',
  padding: theme.spacing(2),
  backgroundColor: 'white',
  borderRadius: '12px',
  boxShadow: '0 2px 12px rgba(0, 0, 0, 0.05)',
  marginBottom: theme.spacing(4),
  '& img': {
    width: '100%',
    height: 'auto',
    borderRadius: '8px',
  },
}));

const GridContainer = styled(Box)(({ theme }) => ({
  display: 'grid',
  gap: theme.spacing(2),
  padding: theme.spacing(2),
  gridTemplateColumns: 'repeat(2, 1fr)',
  '& .full-width': {
    gridColumn: '1 / -1',
  },
  [theme.breakpoints.down('md')]: {
    gridTemplateColumns: '1fr',
    '& > *': {
      gridColumn: '1 / -1',
    },
  },
}));

const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === 'dark' ? '#1A2027' : '#fff',
  ...theme.typography.body2,
  padding: theme.spacing(2),
  textAlign: 'center',
  color: theme.palette.text.secondary,
}));

const ResponsiveImage = styled('img')({
  width: '100%',
  height: 'auto',
  maxWidth: '100%',
  display: 'block',
  margin: '0 auto',
});

const AnalysisPage: React.FC = () => {
  const [selectedTab, setSelectedTab] = React.useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <StyledPaper>
          <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
            Analysis & Results
          </Typography>
          <Tabs 
            value={selectedTab} 
            onChange={handleTabChange} 
            centered
            sx={{
              '& .MuiTab-root': {
                fontSize: '1.1rem',
                fontWeight: 500,
                minWidth: '200px',
              },
            }}
          >
            <Tab icon={<ShowChartIcon />} label="MLP Results" />
            <Tab icon={<TimelineIcon />} label="Time Series Analysis" />
            <Tab icon={<TrendingUpIcon />} label="Comparative Analysis" />
          </Tabs>
        </StyledPaper>

        {selectedTab === 0 && (
          <GridContainer>
            <Box className="full-width">
              <ResultCard>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    MLP Classification Performance
                  </Typography>
                  <Typography paragraph>
                    Our MLP models demonstrated varying effectiveness across different cryptocurrencies and prediction horizons:
                  </Typography>
                  <GridContainer>
                    <Box className="half-width">
                      <Typography variant="h6" gutterBottom>
                        Bitcoin Results
                      </Typography>
                      <ul>
                        <li>Next-day accuracy: 47.9%</li>
                        <li>F1 score: 0.548</li>
                        <li>Recall: 0.719</li>
                        <li>Precision: 0.442</li>
                      </ul>
                    </Box>
                    <Box className="half-width">
                      <Typography variant="h6" gutterBottom>
                        Ethereum Results
                      </Typography>
                      <ul>
                        <li>Next-day accuracy: 45.5%</li>
                        <li>F1 score: 0.606</li>
                        <li>Recall: 0.909</li>
                        <li>Precision: 0.455</li>
                      </ul>
                    </Box>
                  </GridContainer>
                </CardContent>
              </ResultCard>
            </Box>

            <Box className="full-width">
              <ResultCard>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    MLP Regression Performance
                  </Typography>
                  <Typography paragraph>
                    The regression models showed consistent underperformance in R² scores:
                  </Typography>
                  <GridContainer>
                    <Box className="half-width">
                      <Typography variant="h6" gutterBottom>
                        Bitcoin RMSE
                      </Typography>
                      <ul>
                        <li>1-day: 2.75%</li>
                        <li>3-day: 3.91%</li>
                        <li>7-day: 5.11%</li>
                        <li>14-day: 6.68%</li>
                        <li>30-day: 9.90%</li>
                      </ul>
                    </Box>
                    <Box className="half-width">
                      <Typography variant="h6" gutterBottom>
                        Ethereum RMSE
                      </Typography>
                      <ul>
                        <li>1-day: 4.13%</li>
                        <li>3-day: 5.89%</li>
                        <li>7-day: 8.24%</li>
                        <li>14-day: 11.45%</li>
                        <li>30-day: 15.60%</li>
                      </ul>
                    </Box>
                  </GridContainer>
                </CardContent>
              </ResultCard>
            </Box>

            <Box className="full-width">
              <Item>
                <Typography variant="h6" gutterBottom>
                  F1 Score Comparison
                </Typography>
                <ResponsiveImage
                  src={`${process.env.PUBLIC_URL}/F1_comparison.png`}
                  alt="F1 Score Comparison"
                />
              </Item>
            </Box>
          </GridContainer>
        )}

        {selectedTab === 1 && (
          <GridContainer>
            <Box className="full-width">
              <ResultCard>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    Time Series Analysis Results
                  </Typography>
                  <Typography paragraph>
                    Our time series models revealed important insights about cryptocurrency price dynamics:
                  </Typography>
                  <GridContainer>
                    <Box className="half-width">
                      <Typography variant="h6" gutterBottom>
                        ARIMA Model Performance
                      </Typography>
                      <ul>
                        <li>AR(2) model showed superior forecasting capability</li>
                        <li>MAE: 156.23</li>
                        <li>RMSE: 190.60</li>
                        <li>Strong persistence in price movements</li>
                      </ul>
                    </Box>
                    <Box className="half-width">
                      <Typography variant="h6" gutterBottom>
                        GARCH Volatility Analysis
                      </Typography>
                      <ul>
                        <li>High persistence in volatility (α + β ≈ 0.9)</li>
                        <li>Long-lasting effects of volatility shocks</li>
                        <li>Cyclical nature of cryptocurrency volatility</li>
                        <li>Alternating periods of calm and extreme movements</li>
                      </ul>
                    </Box>
                  </GridContainer>
                </CardContent>
              </ResultCard>
            </Box>

            <Box className="full-width">
              <Item>
                <Typography variant="h6" gutterBottom>
                  Bitcoin Price Model Comparison
                </Typography>
                <ResponsiveImage
                  src={`${process.env.PUBLIC_URL}/BTCUSD_model_comparison.png`}
                  alt="Bitcoin Price Comparison"
                />
              </Item>
            </Box>

            <Box className="full-width">
              <Item>
                <Typography variant="h6" gutterBottom>
                  Ethereum Price Model Comparison
                </Typography>
                <ResponsiveImage
                  src={`${process.env.PUBLIC_URL}/ETHUSD_model_comparison.png`}
                  alt="Ethereum Price Comparison"
                />
              </Item>
            </Box>
          </GridContainer>
        )}

        {selectedTab === 2 && (
          <GridContainer>
            <Box className="full-width">
              <ResultCard>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    Comparative Analysis
                  </Typography>
                  <Typography paragraph>
                    Key findings from comparing different modeling approaches:
                  </Typography>
                  <GridContainer>
                    <Box className="half-width">
                      <Typography variant="h6" gutterBottom>
                        Model Strengths
                      </Typography>
                      <ul>
                        <li>MLP: Better at capturing non-linear patterns</li>
                        <li>ARIMA: More interpretable coefficients</li>
                        <li>GARCH: Superior volatility forecasting</li>
                        <li>LSTM: Better sequential dependency modeling</li>
                      </ul>
                    </Box>
                    <Box className="half-width">
                      <Typography variant="h6" gutterBottom>
                        Model Limitations
                      </Typography>
                      <ul>
                        <li>Performance degradation with longer horizons</li>
                        <li>Difficulty in predicting extreme movements</li>
                        <li>Limited adaptability to market regime changes</li>
                        <li>Challenges in integrating external factors</li>
                      </ul>
                    </Box>
                  </GridContainer>
                </CardContent>
              </ResultCard>
            </Box>

            <Box className="full-width">
              <ResultCard>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    Future Improvements
                  </Typography>
                  <Typography paragraph>
                    Recommendations for enhancing model performance:
                  </Typography>
                  <ul>
                    <li>Incorporate alternative data sources (social media, on-chain metrics)</li>
                    <li>Implement market regime detection systems</li>
                    <li>Develop dynamic ensemble methods</li>
                    <li>Add uncertainty quantification through prediction intervals</li>
                    <li>Extend to probability distributions over price movements</li>
                  </ul>
                </CardContent>
              </ResultCard>
            </Box>
          </GridContainer>
        )}
      </Box>
    </Container>
  );
};

export default AnalysisPage; 