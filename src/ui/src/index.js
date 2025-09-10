// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';

import App from './App';

// Suppress ResizeObserver loop error
// This error occurs in some browsers when resizing columns rapidly
const originalConsoleError = console.error;
console.error = (...args) => {
  if (args[0]?.includes?.('ResizeObserver loop')) {
    // Ignore the ResizeObserver loop completed warning
    return;
  }
  originalConsoleError(...args);
};

const container = document.getElementById('root');
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
