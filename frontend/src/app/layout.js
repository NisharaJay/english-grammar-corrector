import "./globals.css";

export const metadata = {
  title: "Grammar Buddy",
  description: "English Grammar Practice Chatbot",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}