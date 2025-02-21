
-- API integration is needed for GitHub integration
CREATE OR REPLACE API INTEGRATION git_api_integration
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com/irelandmi')
  ENABLED = TRUE;


-- Git repository object is similar to external stage
CREATE OR REPLACE GIT REPOSITORY GIT_DEV.GIT.TRADE_SMART
  API_INTEGRATION = git_api_integration
  ORIGIN = 'https://github.com/irelandmi/trade_smart'
  ;

  