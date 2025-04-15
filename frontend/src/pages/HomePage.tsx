import React from 'react';
import { Typography, Paper, Box, Container, Card, CardContent } from '@mui/material';
import { styled } from '@mui/material/styles';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TimelineIcon from '@mui/icons-material/Timeline';
import PsychologyIcon from '@mui/icons-material/Psychology';
import DataObjectIcon from '@mui/icons-material/DataObject';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  marginBottom: theme.spacing(4),
  background: 'linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%)',
  borderRadius: '16px',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
}));

const FeatureCard = styled(Card)(({ theme }) => ({
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

const IconWrapper = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: '64px',
  height: '64px',
  borderRadius: '50%',
  backgroundColor: theme.palette.primary.main,
  marginBottom: theme.spacing(2),
  color: 'white',
}));

const GridContainer = styled(Box)(({ theme }) => ({
  display: 'grid',
  gap: theme.spacing(4),
  gridTemplateColumns: 'repeat(12, 1fr)',
  '& > .full-width': {
    gridColumn: 'span 12',
  },
  '& > .half-width': {
    gridColumn: 'span 12',
    [theme.breakpoints.up('md')]: {
      gridColumn: 'span 6',
    },
  },
}));

const HomePage: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <StyledPaper>
          <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
            Cryptocurrency Price Prediction System
          </Typography>
          <Typography variant="h6" color="text.secondary" paragraph>
            Advanced machine learning models for predicting Bitcoin and Ethereum price movements
          </Typography>
        </StyledPaper>

        <GridContainer>
          <Box className="half-width">
            <FeatureCard>
              <CardContent>
                <IconWrapper>
                  <TrendingUpIcon fontSize="large" />
                </IconWrapper>
                <Typography variant="h5" gutterBottom>
                  Project Overview
                </Typography>
                <Typography paragraph>
                  This project addresses three core challenges in cryptocurrency price prediction:
                </Typography>
                <ul>
                  <li>Modeling rapid price fluctuations (10%+ daily swings)</li>
                  <li>Integrating heterogeneous data sources</li>
                  <li>Capturing temporal dependencies across multiple time horizons</li>
                </ul>
              </CardContent>
            </FeatureCard>
          </Box>

          <Box className="half-width">
            <FeatureCard>
              <CardContent>
                <IconWrapper>
                  <TimelineIcon fontSize="large" />
                </IconWrapper>
                <Typography variant="h5" gutterBottom>
                  Methodology
                </Typography>
                <Typography paragraph>
                  Our dual-model approach combines:
                </Typography>
                <ul>
                  <li>Regression models for magnitude predictions</li>
                  <li>Classification models for directional forecasting</li>
                  <li>Multi-horizon forecasting (1-30 days)</li>
                  <li>Comprehensive data integration</li>
                </ul>
              </CardContent>
            </FeatureCard>
          </Box>

          <Box className="half-width">
            <FeatureCard>
              <CardContent>
                <IconWrapper>
                  <PsychologyIcon fontSize="large" />
                </IconWrapper>
                <Typography variant="h5" gutterBottom>
                  ML Models
                </Typography>
                <Typography paragraph>
                  Advanced machine learning techniques:
                </Typography>
                <ul>
                  <li>Multilayer Perceptron (MLP) Neural Networks</li>
                  <li>ARIMA and GARCH time series models</li>
                  <li>LSTM networks for sequential data</li>
                  <li>Ensemble methods for improved accuracy</li>
                </ul>
              </CardContent>
            </FeatureCard>
          </Box>

          <Box className="half-width">
            <FeatureCard>
              <CardContent>
                <IconWrapper>
                  <DataObjectIcon fontSize="large" />
                </IconWrapper>
                <Typography variant="h5" gutterBottom>
                  Data Sources
                </Typography>
                <Typography paragraph>
                  Comprehensive data collection:
                </Typography>
                <ul>
                  <li>Historical price data from CoinGecko API</li>
                  <li>Technical indicators and market metrics</li>
                  <li>Cryptocurrency-specific development data</li>
                  <li>Market-wide factors and temporal features</li>
                </ul>
              </CardContent>
            </FeatureCard>
          </Box>
        </GridContainer>
      </Box>
    </Container>
  );
};

export default HomePage; 